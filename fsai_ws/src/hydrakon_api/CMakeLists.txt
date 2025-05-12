cmake_minimum_required(VERSION 3.5)
project(hydrakon_api)

# Use both C and C++
enable_language(C CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
  include
)

add_library(fsai_api SHARED
  src/fs-ai_api.c
  src/can.c
)

add_executable(hydrakon_can_node
  src/hydrakon_can.cpp
)

target_link_libraries(hydrakon_can_node fsai_api pthread)
ament_target_dependencies(hydrakon_can_node rclcpp std_msgs)

install(TARGETS
  hydrakon_can_node
  fsai_api
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
)

ament_package()
