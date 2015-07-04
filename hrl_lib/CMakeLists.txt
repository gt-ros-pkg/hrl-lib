cmake_minimum_required(VERSION 2.8.3)
project(hrl_lib)

find_package(catkin REQUIRED COMPONENTS std_srvs geometry_msgs sensor_msgs visualization_msgs tf dynamic_reconfigure hrl_msgs)

catkin_python_setup()

catkin_package(
    DEPENDS std_srvs geometry_msgs sensor_msgs visualization_msgs tf dynamic_reconfigure hrl_msgs
#    CATKIN_DEPENDS # TODO
#    INCLUDE_DIRS # TODO include
    LIBRARIES hrl_lib
)

###########
## Build ##
###########
include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

#install(DIRECTORY plugins
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#)

catkin_install_python(PROGRAMS scripts/keyboard_input
                        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


