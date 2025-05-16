#ifndef HYDRAKON_CAN_BRIDGE_HPP_
#define HYDRAKON_CAN_BRIDGE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/trigger.hpp>

// Custom messages
#include "hydrakon_api/msg/vehicle_command.hpp"
#include "hydrakon_api/msg/wheel_speed.hpp"

// FS-AI API
#include "fs-ai_api.h"

class HydrakonCanBridge : public rclcpp::Node {
public:
  HydrakonCanBridge();

private:
  // Main loop
  void loop();

  // Callbacks
  void commandCallback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg);
  void drivingFlagCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void missionFlagCallback(const std_msgs::msg::Bool::SharedPtr msg);
  bool requestEBS(std_srvs::srv::Trigger::Request::SharedPtr req,
                  std_srvs::srv::Trigger::Response::SharedPtr res);

  // Helpers
  fs_ai_api_handshake_send_bit_e getHandshake();
  fs_ai_api_direction_request_e getDirection();
  fs_ai_api_mission_status_e getMissionStatus();

  void checkTimeout();

  // Publishers
  rclcpp::Publisher<hydrakon_api::msg::VehicleCommand>::SharedPtr vehicle_cmd_pub_;
  rclcpp::Publisher<hydrakon_api::msg::WheelSpeed>::SharedPtr wheel_speed_pub_;

  // Subscribers
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr driving_flag_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr mission_flag_sub_;

  // Services
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr ebs_srv_;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;

  // FS-AI structs
  fs_ai_api_vcu2ai_struct vcu2ai_;
  fs_ai_api_ai2vcu_struct ai2vcu_;
  fs_ai_api_imu_struct imu_;
  fs_ai_api_gps_struct gps_;

  // Control state
  float steering_ = 0.0;
  float torque_ = 0.0;
  float braking_ = 0.0;
  float rpm_ = 0.0;
  bool driving_flag_ = false;
  bool mission_done_ = false;
  double last_cmd_time_ = 0.0;

  // Constants
  const float MAX_STEERING_DEG = 24.0;
  const float MAX_RPM = 4000.0;
  const float MAX_BRAKE = 100.0;
  const float MAX_TORQUE = 195.0;
  const float TIMEOUT_SEC = 0.5;
};

#endif  // HYDRAKON_CAN_BRIDGE_HPP_
