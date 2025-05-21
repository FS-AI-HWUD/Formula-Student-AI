cmake_minimum_required(VERSION 3.8)
project(hydrakon_can)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  # Suppress specific warnings from FS-AI API code
  add_compile_options(-Wno-missing-braces -Wno-missing-field-initializers -Wno-return-type)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VehicleCommand.msg"
  "msg/WheelSpeed.msg"
  DEPENDENCIES std_msgs
)

# Get the include directories for the generated interfaces
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Include directories
include_directories(
  include
  include/hydrakon_can/FS-AI_API/FS-AI_API
)

# Add the executable with all required source files
add_executable(${PROJECT_NAME}_node
  src/hydrakon_can.cpp
  include/hydrakon_can/FS-AI_API/FS-AI_API/fs-ai_api.c
  include/hydrakon_can/FS-AI_API/FS-AI_API/can.c  # Add the can.c file here
)

# Link dependencies for the executable
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
  ackermann_msgs
  tf2
)

# Link generated interfaces
target_link_libraries(${PROJECT_NAME}_node
  "${cpp_typesupport_target}"
)

# Install the executable
install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install other necessary files
install(DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY msg/
  DESTINATION share/${PROJECT_NAME}/msg
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  DIRECTORY include/hydrakon_can/FS-AI_API/FS-AI_API/
  DESTINATION include/hydrakon_can/FS-AI_API
  FILES_MATCHING PATTERN "*.h"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Export dependencies
ament_export_dependencies(
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
  ackermann_msgs
  tf2
  rosidl_default_runtime
)

ament_package()