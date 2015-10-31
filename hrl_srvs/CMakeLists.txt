cmake_minimum_required(VERSION 2.8.3)
project(hrl_srvs)
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)

add_service_files(
  FILES Bool_None.srv FloatArray_None.srv Float_Int.srv Int_None.srv None_Float.srv FloatArray_FloatArray.srv FloatFloat_Int.srv Float_None.srv None_Bool.srv None_Int32.srv FloatArray_Float.srv FloatFloat_None.srv Int_Int.srv None_FloatArray.srv String_None.srv Float_Float.srv String_String.srv
)

generate_messages(
    DEPENDENCIES std_msgs
)

catkin_package(
    DEPENDS 
    CATKIN_DEPENDS message_runtime #std_msgs
    INCLUDE_DIRS 
    LIBRARIES 
)

