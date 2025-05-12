#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

extern "C" {
    #include "hydrakon_api/fs-ai_api.h"  // Your C header
}

class HydrakonCANNode : public rclcpp::Node {
public:
    HydrakonCANNode() : Node("hydrakon_can_node") {
        publisher_ = this->create_publisher<std_msgs::msg::String>("vcu_data", 10);

        if (fs_ai_api_init((char *)"vcan0", 1, 1) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize FS-AI API");
            return;
        }

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&HydrakonCANNode::publish_vcu_data, this)
        );
    }

private:
    void publish_vcu_data() {
        fs_ai_api_vcu2ai data;
        fs_ai_api_vcu2ai_get_data(&data);

        std_msgs::msg::String msg;
        msg.data = "Steering: " + std::to_string(data.VCU2AI_STEER_ANGLE_deg) + " deg";
        publisher_->publish(msg);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HydrakonCANNode>());
    rclcpp::shutdown();
    return 0;
}
