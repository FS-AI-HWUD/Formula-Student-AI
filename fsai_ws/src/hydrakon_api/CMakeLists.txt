cmake_minimum_required(VERSION 3.8)
project(hydrakon_api)

# ROS2 and standard dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Include directories
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Define the executable
add_executable(hydrakon_can_node src/hydrakon_can_node.cpp src/fs-ai_api.c src/can.c)

# C standard for .c files
set_target_properties(hydrakon_can_node PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
)

# Link libraries
ament_target_dependencies(hydrakon_can_node
  rclcpp
  std_msgs
)

# Install target
install(TARGETS hydrakon_can_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
)

ament_package()
