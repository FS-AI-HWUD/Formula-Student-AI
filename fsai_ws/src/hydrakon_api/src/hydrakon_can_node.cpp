#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include "hydrakon_api/fs-ai_api.h"

class HydrakonCANNode : public rclcpp::Node {
public:
    HydrakonCANNode()
    : Node("hydrakon_can_node"),
      axle_speed_(0.0), axle_torque_(0.0), steer_angle_(0.0), brake_pressure_(0.0) {
        
        // Init CAN API
        if (fs_ai_api_init("vcan0", 1, 0) != 0) {
            RCLCPP_FATAL(this->get_logger(), "Failed to init FS-AI API on vcan0");
            rclcpp::shutdown();
        }

        speed_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/vehicle/axle_speed", 10,
            [this](const std_msgs::msg::Float32::SharedPtr msg) {
                axle_speed_ = msg->data;
            });

        torque_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/vehicle/axle_torque", 10,
            [this](const std_msgs::msg::Float32::SharedPtr msg) {
                axle_torque_ = msg->data;
            });

        steer_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/vehicle/steering_angle", 10,
            [this](const std_msgs::msg::Float32::SharedPtr msg) {
                steer_angle_ = msg->data;
            });

        brake_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/vehicle/brake_pressure", 10,
            [this](const std_msgs::msg::Float32::SharedPtr msg) {
                brake_pressure_ = msg->data;
            });

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&HydrakonCANNode::send_data, this));
    }

private:
    void send_data() {
        fs_ai_api_ai2vcu msg = {};
        msg.AI2VCU_AXLE_SPEED_REQUEST_rpm = axle_speed_;
        msg.AI2VCU_AXLE_TORQUE_REQUEST_Nm = axle_torque_;
        msg.AI2VCU_STEER_ANGLE_REQUEST_deg = steer_angle_;
        msg.AI2VCU_BRAKE_PRESS_REQUEST_pct = brake_pressure_;

        msg.AI2VCU_MISSION_STATUS = MISSION_RUNNING;
        msg.AI2VCU_DIRECTION_REQUEST = DIRECTION_FORWARD;
        msg.AI2VCU_ESTOP_REQUEST = ESTOP_NO;
        msg.AI2VCU_HANDSHAKE_SEND_BIT = HANDSHAKE_SEND_BIT_ON;

        RCLCPP_INFO(this->get_logger(), "Sending: steer=%.1f, brake=%.1f, speed=%.1f, torque=%.1f",
                    steer_angle_, brake_pressure_, axle_speed_, axle_torque_);

        fs_ai_api_ai2vcu_set_data(&msg);
    }

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr speed_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr torque_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr steer_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr brake_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    float axle_speed_;
    float axle_torque_;
    float steer_angle_;
    float brake_pressure_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HydrakonCANNode>());
    rclcpp::shutdown();
    return 0;
}
