// hydrakon_can_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include "fs-ai_api.h"

class HydrakonCANNode : public rclcpp::Node {
 public:
  HydrakonCANNode() : Node("hydrakon_can_node") {
    fs_ai_api_init(const_cast<char *>("vcan0"), 1, 0);
    
    steer_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/cmd_steering", 10,
        [this](std_msgs::msg::Float32::SharedPtr msg) { steer_angle_ = msg->data; });

    brake_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/cmd_brake", 10,
        [this](std_msgs::msg::Float32::SharedPtr msg) { brake_pct_ = msg->data; });

    speed_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/cmd_speed", 10,
        [this](std_msgs::msg::Float32::SharedPtr msg) { axle_rpm_ = msg->data; });

    torque_sub_ = this->create_subscription<std_msgs::msg::Float32>(
        "/cmd_torque", 10,
        [this](std_msgs::msg::Float32::SharedPtr msg) { axle_torque_ = msg->data; });

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&HydrakonCANNode::send_data, this));
  }

 private:
  void send_data() {
    fs_ai_api_ai2vcu ai2vcu_data;
    ai2vcu_data.AI2VCU_STEER_ANGLE_REQUEST_deg = steer_angle_;
    ai2vcu_data.AI2VCU_BRAKE_PRESS_REQUEST_pct = brake_pct_;
    ai2vcu_data.AI2VCU_AXLE_SPEED_REQUEST_rpm = axle_rpm_;
    ai2vcu_data.AI2VCU_AXLE_TORQUE_REQUEST_Nm = axle_torque_;

    ai2vcu_data.AI2VCU_ESTOP_REQUEST = ESTOP_NO;
    ai2vcu_data.AI2VCU_DIRECTION_REQUEST = DIRECTION_FORWARD;
    ai2vcu_data.AI2VCU_HANDSHAKE_SEND_BIT = HANDSHAKE_SEND_BIT_ON;
    ai2vcu_data.AI2VCU_MISSION_STATUS = MISSION_SELECTED;

    RCLCPP_INFO(this->get_logger(),
                "Sending: steer=%.2f deg, brake=%.2f %%, rpm=%.2f, torque=%.2f Nm",
                steer_angle_, brake_pct_, axle_rpm_, axle_torque_);

    fs_ai_api_ai2vcu_set_data(&ai2vcu_data);
  }

  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr steer_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr brake_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr speed_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr torque_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  float steer_angle_ = 0.0;
  float brake_pct_ = 0.0;
  float axle_rpm_ = 0.0;
  float axle_torque_ = 0.0;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HydrakonCANNode>());
  rclcpp::shutdown();
  return 0;
}
