#!/usr/bin/env python
# Software License Agreement (BSD License)
#
#  Copyright (c) 2011, Georgia Institute of Technology
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#  * Neither the name of the Georgia Institute of Technology nor the names of
#     its contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  'AS IS' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import roslib; roslib.load_manifest('tabletop_pushing')
import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
import tf
import tf.transformations as tf_trans
import sys

class TestNode:

    def __init__(self):
       rospy.init_node('test_node', log_level=rospy.DEBUG)

    def talk(self):
        # pub = rospy.Publisher('command', PoseStamped)
        pub = rospy.Publisher('r_cart/command_pose', PoseStamped)
        rospy.loginfo('created the publisher obj')
        
        pose = PoseStamped()
        pose.header.frame_id = '/torso_lift_link'
        pose.header.stamp = rospy.Time(0)
        pose.pose.position.x = 0.3
        pose.pose.position.y = -0.2
        pose.pose.position.z = -0.20
        pose.pose.orientation.x = 1
        pose.pose.orientation.y = 0 
        pose.pose.orientation.z = 0.
        pose.pose.orientation.w = 0.
        while not rospy.is_shutdown():
            	rospy.loginfo('Publishing following message: %s'%pose)
            	pub.publish(pose)
            	rospy.sleep(2.0) 

if __name__ == '__main__':
    try:
        node = TestNode()
        node.talk()
    except rospy.ROSInterruptException: pass
