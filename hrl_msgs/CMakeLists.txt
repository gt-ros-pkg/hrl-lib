cmake_minimum_required(VERSION 2.8.3)
project(hrl_msgs)
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)

add_message_files(
  DIRECTORY msg
  FILES FloatArray.msg FloatArrayBare.msg StringArray.msg FloatMatrix.msg IntArray.msg IntArrayBare.msg MeshAttrBare.msg
)

generate_messages(
    DEPENDENCIES std_msgs
)

catkin_package(
    DEPENDS roscpp rospy
    CATKIN_DEPENDS message_runtime std_msgs
    INCLUDE_DIRS 
    LIBRARIES 
)

