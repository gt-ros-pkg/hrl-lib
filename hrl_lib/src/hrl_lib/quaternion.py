#
# Copyright (c) 2014, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#  \author Daehyung Park (Healthcare Robotics Lab, Georgia Tech.)

import roslib; roslib.load_manifest('hrl_lib')

import os, sys
import copy
import numpy as np
import time
import math

import tf.transformations as tft

#copied from manipulation stack
#angle between two quaternions (as lists)
def quat_angle(quat1, quat2):
    dot = sum([x*y for (x,y) in zip(quat1, quat2)])
    if dot > 1.:
        dot = 1.
    if dot < -1.:
        dot = -1.
    angle = 2*math.acos(math.fabs(dot))
    return angle     

# Return a co-distance matrix between X and Y.
# X and Y are a set of quaternions.
def quat_angles( X, Y ):

    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )

    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            d[i,j] = quat_angle(x,y)

    return d

# Return a quaternion, which is a negation of input quaternion.
def quat_inv_sign(q):
    return q* (-1.0)    

# Return a normalized quaternion
def quat_normal(q):
    mag = np.sqrt(np.sum(q*q))
    return q/mag

# Return true or false whether two quaternion is on a same hemisphere.
def AreQuaternionClose(q1,q2):
    
    dot = np.sum(q1*q2)
    
    if dot < 0.0:
        return False
    else:
        return True

# Return an averaged quaternion
def quat_avg( X ):

    n,m = X.shape
    cumulative_x = X[0]

    for i,x in enumerate(X):

        if i==0: continue
        
        new_x = copy.copy(x)
        if not AreQuaternionClose(new_x,X[0]):
            new_x = quat_inv_sign(new_x)

        cumulative_x += new_x

    cumulative_x /= float(n)

    return quat_normal(cumulative_x)

    
# Return n numbers of uniform random quaternions.    
def quat_random( n ):

    u1 = random.random()
    u2 = random.random()
    u3 = random.random()

    X = np.array([np.sqrt(1-u1)*np.sin(2.0*np.pi*u2),
                  np.sqrt(1-u1)*np.cos(2.0*np.pi*u2),
                  np.sqrt(u1)*np.sin(2.0*np.pi*u3),
                  np.sqrt(u1)*np.cos(2.0*np.pi*u3)])

    count = 1
    while True:
        if count == n: break
        else: count += 1

        u1 = random.random()
        u2 = random.random()
        u3 = random.random()

        X = np.vstack([X, np.array([np.sqrt(1-u1)*np.sin(2.0*np.pi*u2),
                                    np.sqrt(1-u1)*np.cos(2.0*np.pi*u2),
                                    np.sqrt(u1)*np.sin(2.0*np.pi*u3),
                                    np.sqrt(u1)*np.cos(2.0*np.pi*u3)])])

    return X

def quat_to_angle_and_axis( q ):

    mat = tft.quaternion_matrix(q)
    angle, direction, point = tft.rotation_from_matrix(mat)
                
    return angle, direction
