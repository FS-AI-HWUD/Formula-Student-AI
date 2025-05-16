cmake_minimum_required(VERSION 3.8)
project(hydrakon_api)

# Use C++17 standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# Include header paths
include_directories(
  include
  ${PROJECT_SOURCE_DIR}/include/hydrakon_api
)

# Source files
set(SOURCES
  # src/hydrakon_can_node.cpp
  src/fs-ai_api.c
  src/can.c
  src/hydrakon_can_bridge.cpp
)
ament_export_dependencies(rosidl_default_runtime)

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VehicleCommand.msg"
  "msg/WheelSpeed.msg"
)

# Build executable
# add_executable(hydrakon_can_node ${SOURCES})
# --- Executable ---
add_executable(hydrakon_can_bridge_node ${SOURCES})

ament_target_dependencies(hydrakon_can_bridge_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  ackermann_msgs
  std_srvs
)
# # Link dependencies
# ament_target_dependencies(hydrakon_can_node
#   rclcpp
#   std_msgs
# )

# # Install target
# install(TARGETS hydrakon_can_node
#   DESTINATION lib/${PROJECT_NAME}
# )

install(TARGETS hydrakon_can_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)
# --- Link generated messages ---
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

target_link_libraries(hydrakon_can_bridge_node
  ${cpp_typesupport_target}
)

# Install all headers (including can.h and fs-ai_api.h)
install(DIRECTORY include/
  DESTINATION include/
)

# ament package
ament_package()
