cmake_minimum_required(VERSION 2.8.3)
project(hrl_lib)

find_package(catkin REQUIRED COMPONENTS std_srvs geometry_msgs sensor_msgs visualization_msgs tf dynamic_reconfigure hrl_msgs)

catkin_python_setup()

catkin_package(
    DEPENDS 
    CATKIN_DEPENDS std_srvs geometry_msgs sensor_msgs visualization_msgs tf dynamic_reconfigure hrl_msgs
    INCLUDE_DIRS 
    LIBRARIES
)

catkin_install_python(PROGRAMS 
                      scripts/keyboard_input 
                      scripts/save_dynamic_config
                      scripts/load_dynamic_config
                    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


