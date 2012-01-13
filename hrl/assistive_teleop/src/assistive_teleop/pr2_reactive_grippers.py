#!/usr/bin/python

import roslib; roslib.load_manifest('assistive_teleop')
import rospy

import actionlib
from pr2_controllers_msgs.msg import (Pr2GripperCommandGoal, Pr2GripperCommand,
                                      Pr2GripperCommandAction)
from pr2_gripper_sensor_msgs.msg import (
    PR2GripperGrabAction, PR2GripperGrabGoal, PR2GripperGrabCommand,
    PR2GripperReleaseAction, PR2GripperReleaseGoal, PR2GripperReleaseCommand,
    PR2GripperEventDetectorCommand
    )

TEST = True

class PR2Gripper():
    def __init__(self, arm):
        #TODO: Adapt smoothly to changing gripper controller
        self.arm = arm

        ####ACTION CLIENTS####
        self.grab_ac = actionlib.SimpleActionClient(self.arm[0]+
                                            '_gripper_sensor_controller/grab',
                                            PR2GripperGrabAction)
        rospy.loginfo("Waiting for " + self.arm + " gripper grab server")
        if self.grab_ac.wait_for_server(rospy.Duration(50)):
            rospy.loginfo("Found " + self.arm + " gripper grab  server")
        else:
            rospy.logwarn("Cannot find " + self.arm + " gripper grab server")
       
        self.release_ac = actionlib.SimpleActionClient(self.arm[0]+
                                        '_gripper_sensor_controller/release',
                                        PR2GripperReleaseAction)
        rospy.loginfo("Waiting for " + self.arm + " gripper release server")
        if self.grab_ac.wait_for_server(rospy.Duration(50)):
            rospy.loginfo("Found "+self.arm+" gripper release server")
        else:
            rospy.logwarn("Cannot find "+self.arm+" gripper release server")

        self.gripper_action_ac = actionlib.SimpleActionClient(self.arm[0]+
                                '_gripper_sensor_controller/gripper_action',
                                Pr2GripperCommandAction)
        rospy.loginfo("Waiting for "+self.arm+" gripper action server")
        if self.gripper_action_ac.wait_for_server(rospy.Duration(50)):
            rospy.loginfo("Found "+self.arm+" gripper action server")
        else:
            rospy.logwarn("Cannot find "+self.arm+" gripper action server")
    
    def grab(self, gain=0.03, blocking=False, block_timeout=20):
        print "Performing Gripper Grab"
        grab_goal = PR2GripperGrabGoal(PR2GripperGrabCommand(gain))
        self.grab_ac.send_goal(grab_goal)
        if blocking:
            return self.grab_ac.wait_for_result(rospy.Duration(block_timeout))

    def release(self, event=0, acc_thresh=5, slip_thresh=0.01,
                blocking=False, block_timeout=20):
        print "Performing Gripper Release"
        release_event=PR2GripperEventDetectorCommand()
        release_event.trigger_conditions = event
        release_event.acceleration_trigger_magnitude = acc_thresh
        release_event.slip_trigger_magnitude = slip_thresh
        release_command = PR2GripperReleaseCommand(release_event)
        self.release_ac.send_goal(PR2GripperReleaseGoal(release_command))
        if blocking:
            return self.release_ac.wait_for_result(rospy.Duration(block_timeout))

    def gripper_action(self, position, max_effort=-1, 
                        blocking=False, block_timeout=20.0):
         print "Performing Gripper Action"
         command = Pr2GripperCommand(position, max_effort)
         goal = Pr2GripperCommandGoal(command)
         self.gripper_action_ac.send_goal(goal)
         if blocking:
             return self.gripper_action_ac.wait_for_result(rospy.Duration(block_timeout))
        
if __name__=='__main__':
    rospy.init_node('gripper_sensor_intermediary')
    if TEST:
        gripper = PR2Gripper('right')
        print "Initialized!"
        print "Attempting to close to 0.01"
        gripper.gripper_action(0.01,blocking=True)
        print "Done closing, performing grab"
        gripper.grab(blocking=True, block_timeout=2)
        print "Done Grabbing, performing close to 0.005"
        gripper.gripper_action(0.005,blocking=True)
        print "Done closing, performing release"
        gripper.release(blocking=True)
        print "Done Realeasing: Test Complete"

    while not rospy.is_shutdown():
        rospy.spin()
