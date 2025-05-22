cmake_minimum_required(VERSION 3.5)
project(hydrakon_can)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/WheelSpeed.msg"
  "msg/VehicleCommand.msg"
  "msg/CanState.msg"
  DEPENDENCIES std_msgs
)

# Make FS-AI-API
add_custom_target(fs-ai-api ALL
  COMMAND $(MAKE) -C ${PROJECT_SOURCE_DIR}/include/hydrakon_can/FS-AI_API/FS-AI_API -f makefile
)

# Create hydrakon_can_node executable
include_directories(include include/hydrakon_can/FS-AI_API/FS-AI_API)
add_executable(hydrakon_can_node src/hydrakon_can.cpp)
add_dependencies(hydrakon_can_node fs-ai-api ${PROJECT_NAME})
target_link_libraries(hydrakon_can_node ${PROJECT_SOURCE_DIR}/include/hydrakon_can/FS-AI_API/FS-AI_API/fs-ai_api.a)
ament_target_dependencies(hydrakon_can_node rclcpp tf2 tf2_ros std_srvs sensor_msgs std_msgs ackermann_msgs)

# Link with generated interfaces
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(hydrakon_can_node "${cpp_typesupport_target}")

# Install executable
install(TARGETS hydrakon_can_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files.
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()