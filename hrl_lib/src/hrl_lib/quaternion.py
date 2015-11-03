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

# NOTE: tf's quaternion order is [x,y,z,w]

import copy
import random
import numpy as np
import math

# ROS & Public library
import tf.transformations as tft
from geometry_msgs.msg import Quaternion

#copied from manipulation stack
#angle between two quaternions (as lists)
def quat_angle(quat1, quat2):

    if type(quat1) == Quaternion:
        quat1 = [quat1.x, quat1.y, quat1.z, quat1.w]
    if type(quat2) == Quaternion:
        quat2 = [quat2.x, quat2.y, quat2.z, quat2.w]

    dot = math.fabs(sum([x*y for (x,y) in zip(quat1, quat2)]))
    if dot > 1.0: dot = 1.0
    angle = 2*math.acos( dot )
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
    return q/np.linalg.norm(q)

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

    # Quaternions ix+jy+kz+w are represented as [x, y, z, w].
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


# quat_mean: a quaternion(xyzw) that is the center of gaussian distribution
# n:
# stdDev: a vector (4x1) that describes the standard deviations of the distribution
#         along axis(xyz) and angle
# Return n numbers of QuTem quaternions (gaussian distribution).
def quat_QuTem( quat_mean, n, stdDev ):

    # Gaussian random quaternion
    x = (np.array([np.random.normal(0., 1., n)]).T *stdDev[0]*stdDev[0])
    y = (np.array([np.random.normal(0., 1., n)]).T *stdDev[1]*stdDev[1])
    z = (np.array([np.random.normal(0., 1., n)]).T *stdDev[2]*stdDev[2])

    mag = np.zeros((n,1))
    for i in xrange(len(x)):
        mag[i,0] = np.sqrt([x[i,0]**2+y[i,0]**2+z[i,0]**2])

    axis  = np.hstack([x/mag, y/mag, z/mag])
    ## angle = np.array([np.random.normal(0., stdDev[3]**2.0, n)]).T
    angle = np.zeros([len(x),1])
    for i in xrange(len(x)):
        rnd = 0.0
        while True:
            rnd = np.random.normal(0.,1.)
            if rnd <= np.pi and rnd > -np.pi:
                break
        angle[i,0] = rnd + np.pi

    # Convert the gaussian axis and angle distribution to unit quaternion distribution
    # angle should be limited to positive range...
    s = np.sin(angle / 2.0);
    quat_rnd = np.hstack([axis*s, np.cos(angle/2.0)])

    # Multiplication with mean quat
    q = np.zeros((n,4))
    for i in xrange(len(x)):
        q[i,:] = tft.quaternion_multiply(quat_mean, quat_rnd[i,:])

    return q


# Return an axis and angle converted from a quaternion.
def quat_to_angle_and_axis( q ):

    ## mat = tft.quaternion_matrix(q)
    ## angle, direction, point = tft.rotation_from_matrix(mat)

    ## # tf has some numerical error around pi.
    ## if (abs(angle)-np.pi)**2 < 0.001 and angle < 0.0:
    ##     print "fix error", angle
    ##     angle = abs(angle)
    ##     direction *= -1.0

    ## print q, angle, direction

    ## Reference: http://www.euclideanspace.com
    if abs(q[3]) > 1.0: q /= np.linalg.norm(q)
    angle = 2.0 * math.acos(q[3]) # 0~2pi
    s = np.sqrt(1-q[3]*q[3]) # assuming quaternion normalised then w is less than 1, so term always positive.
    if s < 0.001: # test to avoid divide by zero, s is always positive due to sqrt
        #if s close to zero then direction of axis not important
        # if it is important that axis is normalised then replace with x=1; y=z=0;
        direction = np.array([q[0],q[1],q[2]])
        print "s is closed to singular"
    else:
        # normalise axis
        direction = np.array([q[0],q[1],q[2]])/s

    if direction[0] > 0.0:
        print angle, q[3], direction,s

    return angle, direction


# Return a rotation matrix from a quaternion
def quat2rot(quat):
    # quaternion [w,x,y,z]
    rot = np.matrix([[1 - 2*quat.y*quat.y - 2*quat.z*quat.z,	2*quat.x*quat.y - 2*quat.z*quat.w,      2*quat.x*quat.z + 2*quat.y*quat.w],
                    [2*quat.x*quat.y + 2*quat.z*quat.w, 	    1 - 2*quat.x*quat.x - 2*quat.z*quat.z, 	2*quat.y*quat.z - 2*quat.x*quat.w],
                    [2*quat.x*quat.z - 2*quat.y*quat.w, 	    2*quat.y*quat.z + 2*quat.x*quat.w, 	    1 - 2*quat.x*quat.x - 2*quat.y*quat.y]])
    return rot


def euler2quat(z=0, y=0, x=0):
    ''' Return quaternion corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    We can derive this formula in Sympy using:

    1. Formula giving quaternion corresponding to rotation of theta radians
       about arbitrary axis:
       http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
       theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
       http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
       formulae from 2.) to give formula for combined rotations.
    '''
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = np.cos(z)
    sz = np.sin(z)
    cy = np.cos(y)
    sy = np.sin(y)
    cx = np.cos(x)
    sx = np.sin(x)
    return np.array([cx*sy*sz + cy*cz*sx,
                     cx*cz*sy - sx*cy*sz,
                     cx*cy*sz + sx*cz*sy,
                     cx*cy*cz - sx*sy*sz])


# Will be removed. Use tft.quaternion_multiply.
# quaternoin [x,y,z,w]
def quat_quat_mult(q1, q2):
    [x1, y1, z1, w1] = q1
    [x2, y2, z2, w2] = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([x, y, z, w])


# t=0, then qm=qa
def slerp(qa, qb, t):

    if type(qa) == np.ndarray or type(qa) == list:
        q1 = Quaternion()
        q1.x = qa[0]
        q1.y = qa[1]
        q1.z = qa[2]
        q1.w = qa[3]
    else: q1 = qa
    if type(qb) == np.ndarray or type(qb) == list:
        q2 = Quaternion()
        q2.x = qb[0]
        q2.y = qb[1]
        q2.z = qb[2]
        q2.w = qb[3]
    else: q2 = qb
    
	# quaternion to return
    qm = Quaternion()

	# Calculate angle between them.
    cosHalfTheta = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z
	# if q1=q2 or q1=-q2 then theta = 0 and we can return q1
    if abs(cosHalfTheta) >= 1.0:
        qm.w = q1.w;qm.x = q1.x;qm.y = q1.y;qm.z = q1.z
        if type(qa) == np.ndarray or type(qa) == list: return qa
        else: return qm

    # shortest path
    if cosHalfTheta < 0.0:
        q2.w *= -1.0
        q2.x *= -1.0
        q2.y *= -1.0
        q2.z *= -1.0
        
        cosHalfTheta *= -1.0

    # Calculate temporary values.
    halfTheta = np.arccos(cosHalfTheta)
    sinHalfTheta = np.sqrt(1.0 - np.cos(halfTheta)*np.cos(halfTheta))
    
    # if theta = 180 degrees then result is not fully defined
    # we could rotate around any axis normal to q1 or q2
    if abs(sinHalfTheta) < 0.001: # fabs is floating point absolute
        qm.w = (q1.w * 0.5 + q2.w * 0.5)
        qm.x = (q1.x * 0.5 + q2.x * 0.5)
        qm.y = (q1.y * 0.5 + q2.y * 0.5)
        qm.z = (q1.z * 0.5 + q2.z * 0.5)
        if type(qa) == np.ndarray or type(qa) == list:
            return np.array([qm.x, qm.y, qm.z, qm.w])
        else: return qm

    ratioA = np.sin((1 - t) * halfTheta) / sinHalfTheta
    ratioB = np.sin(t * halfTheta) / sinHalfTheta

    #calculate Quaternion.
    qm.w = (q1.w * ratioA + q2.w * ratioB)
    qm.x = (q1.x * ratioA + q2.x * ratioB)
    qm.y = (q1.y * ratioA + q2.y * ratioB)
    qm.z = (q1.z * ratioA + q2.z * ratioB)
    
    #mag = np.sqrt(qm.w**2+qm.x**2+qm.y**2+qm.z**2)
    #print mag
    ## qm.w /= mag
    ## qm.x /= mag
    ## qm.y /= mag
    ## qm.z /= mag    
    if (type(qa) == np.ndarray and type(qb) == np.ndarray) or \
        (type(qa) == list and type(qb) == list):
        return np.array([qm.x, qm.y, qm.z, qm.w])
    else:
        return qm
