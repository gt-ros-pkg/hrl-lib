#! /usr/bin/python

import sys
import numpy as np
import copy
from threading import Lock

import roslib
roslib.load_manifest('hrl_pr2_arms')
roslib.load_manifest('hrl_rfh_summer_2011')
roslib.load_manifest('hrl_rfh_fall_2011')
roslib.load_manifest('pr2_traj_playback')
import rospy

import smach
import actionlib
from smach_ros import SimpleActionState, ServiceState, IntrospectionServer
from std_srvs.srv import Empty, EmptyResponse

from hrl_rfh_fall_2011.sm_register_head_ellipse import SMEllipsoidRegistration
from hrl_trajectory_playback.srv import TrajPlaybackSrv, TrajPlaybackSrvRequest
from pr2_controllers_msgs.msg import SingleJointPositionAction, SingleJointPositionGoal
from pr2_controllers_msgs.msg import Pr2GripperCommandAction, Pr2GripperCommandGoal
from hrl_rfh_fall_2011.sm_topic_monitor import TopicMonitor
from hrl_pr2_arms.pr2_arm import create_pr2_arm, PR2ArmJTransposeTask
from hrl_pr2_arms.pr2_controller_switcher import ControllerSwitcher
from pr2_traj_playback.msg import TrajectoryPlayAction, TrajectoryPlayGoal

def get_goal_template():
    traj = TrajectoryPlayGoal()
    traj.mode = traj.SETUP_AND_TRAJ
    traj.reverse = False
    traj.setup_velocity = 0.1
    traj.traj_rate_mult = 0.8
    return traj

class SetupArmsShaving():
    def __init__(self):
        self.lock = Lock()
        self.ctrl_switcher = ControllerSwitcher()
        self.traj_playback_l = actionlib.SimpleActionClient('/trajectory_playback_l', TrajectoryPlayAction)
        self.traj_playback_r = actionlib.SimpleActionClient('/trajectory_playback_r', TrajectoryPlayAction)
        self.torso_sac = actionlib.SimpleActionClient('torso_controller/position_joint_action',
                                                      SingleJointPositionAction)
        self.gripper_sac = actionlib.SimpleActionClient(
                                          '/l_gripper_controller/gripper_action',
                                          Pr2GripperCommandAction)
        self.torso_sac.wait_for_server()
        self.traj_playback_l.wait_for_server()
        self.traj_playback_r.wait_for_server()
        self.gripper_sac.wait_for_server()
        rospy.loginfo("[setup_arms_shaving] SetupArmsShaving ready.")

    def adjust_torso(self):
        # move torso up
        tgoal = SingleJointPositionGoal()
        tgoal.position = 0.040  # all the way up is 0.300
        tgoal.min_duration = rospy.Duration( 2.0 )
        tgoal.max_velocity = 1.0
        self.torso_sac.send_goal_and_wait(tgoal)

    def close_gripper(self):
        ggoal = Pr2GripperCommandGoal()
        ggoal.command.position = 0.0
        ggoal.command.max_effort = 10.0
        self.gripper_sac.send_goal_and_wait(ggoal)

    def setup_task_controller(self):
        self.ctrl_switcher.carefree_switch('l', '%s_cart_jt_task', 
                                           "$(find hrl_rfh_fall_2011)/params/l_jt_task_shaver45.yaml") 
        rospy.sleep(0.3)
        self.arm = create_pr2_arm('l', PR2ArmJTransposeTask, 
                                  controller_name='%s_cart_jt_task', 
                                  end_link="%s_gripper_shaver45_frame")
        shaving_side = rospy.get_param("shaving_side")
        if shaving_side == "r":
            setup_angles = [0, 0, np.pi/2, -np.pi/2, -np.pi, -np.pi/2, -np.pi/2]
            self.arm.set_posture(setup_angles)
        else:
            self.arm.set_posture(self.arm.get_joint_angles())
        self.arm.set_gains([200, 800, 800, 80, 80, 80], [15, 15, 15, 1.2, 1.2, 1.2])
        rospy.sleep(0.3)

    def move_to_setup(self):
        shaving_side = rospy.get_param("/shaving_side")
        print shaving_side
        assert(shaving_side in ["r", "l"])

        l_traj = get_goal_template()
        print ("$(find hrl_rfh_fall_2011)/data/l_arm_shaving_setup_%s.pkl" %
                           shaving_side)
        l_traj.filepath = ("$(find hrl_rfh_fall_2011)/data/l_arm_shaving_setup_%s.pkl" %
                           shaving_side)

        r_traj = get_goal_template()
        r_traj.filepath = ("$(find hrl_rfh_fall_2011)/data/r_arm_shaving_setup_%s.pkl" %
                           shaving_side)

        self.traj_playback_l.send_goal(l_traj)
        self.traj_playback_r.send_goal(r_traj)
        self.traj_playback_l.wait_for_result()
        self.traj_playback_r.wait_for_result()
    

    def run(self, req):
        ############################################################
        self.lock.acquire(False)

        self.close_gripper()
        self.adjust_torso()
        self.move_to_setup()
        self.setup_task_controller()

        self.lock.release()
        ############################################################

        return EmptyResponse()


def main():
    rospy.init_node("setup_arms_shaving")
    assert(len(sys.argv) > 1)
    sas = SetupArmsShaving()
    if sys.argv[1] == "-s":
        rospy.Service("/setup_arms_shaving", Empty, sas.run)
        rospy.spin()
    else:
        sas.run(None)
        
if __name__ == "__main__":
    main()