#include "hydrakon_can_bridge.hpp"

HydrakonCanBridge::HydrakonCanBridge() : Node("hydrakon_can_bridge") {
  fs_ai_api_init(const_cast<char *>("vcan0"), 1, 0);

  cmd_sub_ = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
      "/hydrakon_can/cmd", 10, std::bind(&HydrakonCanBridge::commandCallback, this, std::placeholders::_1));

  driving_flag_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "/hydrakon_can/driving_flag", 10, std::bind(&HydrakonCanBridge::drivingFlagCallback, this, std::placeholders::_1));

  mission_flag_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "/hydrakon_can/mission_done", 10, std::bind(&HydrakonCanBridge::missionFlagCallback, this, std::placeholders::_1));

  vehicle_cmd_pub_ = this->create_publisher<hydrakon_api::msg::VehicleCommand>("/hydrakon_can/vehicle_command", 10);
  wheel_speed_pub_ = this->create_publisher<hydrakon_api::msg::WheelSpeed>("/hydrakon_can/wheel_speeds", 10);

  ebs_srv_ = this->create_service<std_srvs::srv::Trigger>("/hydrakon_can/ebs", std::bind(&HydrakonCanBridge::requestEBS, this, std::placeholders::_1, std::placeholders::_2));

  timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&HydrakonCanBridge::loop, this));
}

void HydrakonCanBridge::commandCallback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) {
  float accel = msg->drive.acceleration;
  steering_ = msg->drive.steering_angle * 180.0 / M_PI;
  steering_ = std::max(std::min(steering_, MAX_STEERING_DEG), -MAX_STEERING_DEG);

  if (accel > 0.0) {
    torque_ = std::min(accel * MAX_TORQUE, MAX_TORQUE);
    braking_ = 0.0;
    rpm_ = MAX_RPM;
  } else {
    torque_ = 0.0;
    braking_ = std::min(-accel * MAX_BRAKE, MAX_BRAKE);
    rpm_ = 0.0;
  }

  last_cmd_time_ = this->now().seconds();
}

void HydrakonCanBridge::drivingFlagCallback(const std_msgs::msg::Bool::SharedPtr msg) {
  driving_flag_ = msg->data;
  if (driving_flag_ && last_cmd_time_ == 0.0)
    last_cmd_time_ = this->now().seconds();
  else if (!driving_flag_)
    last_cmd_time_ = 0.0;
}

void HydrakonCanBridge::missionFlagCallback(const std_msgs::msg::Bool::SharedPtr msg) {
  mission_done_ = msg->data;
}

bool HydrakonCanBridge::requestEBS(std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr res) {
  ai2vcu_.AI2VCU_ESTOP_REQUEST = fs_ai_api_estop_request_e::ESTOP_YES;
  res->success = true;
  RCLCPP_WARN(this->get_logger(), "EBS Triggered!");
  return true;
}

void HydrakonCanBridge::loop() {
  // Read data from vehicle
  fs_ai_api_vcu2ai_get_data(&vcu2ai_);
  // fs_ai_api_gps_get_data(&gps_);
  // fs_ai_api_imu_get_data(&imu_);

  // Compose outbound struct
  ai2vcu_.AI2VCU_STEER_ANGLE_REQUEST_deg = steering_;
  ai2vcu_.AI2VCU_BRAKE_PRESS_REQUEST_pct = braking_;
  ai2vcu_.AI2VCU_AXLE_TORQUE_REQUEST_Nm = torque_;
  ai2vcu_.AI2VCU_AXLE_SPEED_REQUEST_rpm = rpm_;
  ai2vcu_.AI2VCU_HANDSHAKE_SEND_BIT = getHandshake();
  ai2vcu_.AI2VCU_DIRECTION_REQUEST = getDirection();
  ai2vcu_.AI2VCU_MISSION_STATUS = getMissionStatus();

  fs_ai_api_ai2vcu_set_data(&ai2vcu_);

  // Publish vehicle command
  hydrakon_api::msg::VehicleCommand cmd_msg;
  cmd_msg.steering = steering_;
  cmd_msg.torque = torque_;
  cmd_msg.braking = braking_;
  cmd_msg.rpm = rpm_;
  cmd_msg.handshake = getHandshake();
  cmd_msg.direction = getDirection();
  cmd_msg.ebs = ai2vcu_.AI2VCU_ESTOP_REQUEST;
  cmd_msg.mission_status = getMissionStatus();
  vehicle_cmd_pub_->publish(cmd_msg);

  // Publish wheel speeds
  hydrakon_api::msg::WheelSpeed wheel_msg;
  wheel_msg.steering = -vcu2ai_.VCU2AI_STEER_ANGLE_deg * M_PI / 180.0f;
  wheel_msg.lf_speed = vcu2ai_.VCU2AI_FL_WHEEL_SPEED_rpm;
  wheel_msg.rf_speed = vcu2ai_.VCU2AI_FR_WHEEL_SPEED_rpm;
  wheel_msg.lb_speed = vcu2ai_.VCU2AI_RL_WHEEL_SPEED_rpm;
  wheel_msg.rb_speed = vcu2ai_.VCU2AI_RR_WHEEL_SPEED_rpm;
  wheel_speed_pub_->publish(wheel_msg);

  if (driving_flag_) checkTimeout();
}

fs_ai_api_handshake_send_bit_e HydrakonCanBridge::getHandshake() {
  return (vcu2ai_.VCU2AI_HANDSHAKE_RECEIVE_BIT == HANDSHAKE_RECEIVE_BIT_ON)
    ? HANDSHAKE_SEND_BIT_ON
    : HANDSHAKE_SEND_BIT_OFF;
}

fs_ai_api_direction_request_e HydrakonCanBridge::getDirection() {
  return (vcu2ai_.VCU2AI_AS_STATE == AS_DRIVING && driving_flag_)
    ? DIRECTION_FORWARD
    : DIRECTION_NEUTRAL;
}

fs_ai_api_mission_status_e HydrakonCanBridge::getMissionStatus() {
  switch (vcu2ai_.VCU2AI_AS_STATE) {
    case AS_OFF:
      return MISSION_NOT_SELECTED;
    case AS_READY:
      return driving_flag_ ? MISSION_RUNNING : MISSION_SELECTED;
    case AS_DRIVING:
      return mission_done_ ? MISSION_FINISHED : MISSION_RUNNING;
    case AS_FINISHED:
      return MISSION_FINISHED;
    default:
      return MISSION_NOT_SELECTED;
  }
}


void HydrakonCanBridge::checkTimeout() {
  if (this->now().seconds() - last_cmd_time_ > TIMEOUT_SEC) {
    RCLCPP_ERROR(this->get_logger(), "CMD timeout, triggering EBS");
    ai2vcu_.AI2VCU_ESTOP_REQUEST = fs_ai_api_estop_request_e::ESTOP_YES;
  }
}

// --- Main ---
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HydrakonCanBridge>());
  rclcpp::shutdown();
  return 0;
}
