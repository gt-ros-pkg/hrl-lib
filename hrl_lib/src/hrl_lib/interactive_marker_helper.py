import interactive_markers.interactive_marker_server as ims
import std_msgs.msg as stdm

def feedback_to_string(ftype):
    names = ['keep_alive', 'pose_update', 
             'menu_select', 'button_click',
             'mouse_down', 'mouse_up']
    fb = ims.InteractiveMarkerFeedback
    consts = [fb.KEEP_ALIVE, fb.POSE_UPDATE,
                fb.MENU_SELECT, fb.BUTTON_CLICK,
                fb.MOUSE_DOWN, fb.MOUSE_UP]

    for n, value in zip(names, consts):
        if ftype == value:
            return n

    return 'invalid type'


def interactive_marker(name, pose, scale):
    int_marker = ims.InteractiveMarker()
    int_marker.header.frame_id = "/map"
    int_marker.pose.position.x = pose[0][0]
    int_marker.pose.position.y = pose[0][1]
    int_marker.pose.position.z = pose[0][2]
    int_marker.pose.orientation.x = pose[1][0]
    int_marker.pose.orientation.y = pose[1][1]
    int_marker.pose.orientation.z = pose[1][2]
    int_marker.pose.orientation.w = pose[1][3]
    
    int_marker.scale = scale
    int_marker.name = name
    int_marker.description = name
    return int_marker

def make_rviz_marker(scale):
    marker = ims.Marker()
    marker.type = ims.Marker.SPHERE
    marker.scale.x = scale * 0.45
    marker.scale.y = scale * 0.45
    marker.scale.z = scale * 0.45
    marker.color = stdm.ColorRGBA(.5,.5,.5,1)
    return marker

def make_sphere_control(name, scale):
    control =  ims.InteractiveMarkerControl()
    control.name = name + '_sphere'
    control.always_visible = True
    control.markers.append(make_rviz_marker(scale))
    control.interaction_mode = ims.InteractiveMarkerControl.BUTTON
    return control

def make_control_marker(orientation=[0,0,0,1.]):
    control = ims.InteractiveMarkerControl()
    control.orientation.x = orientation[0]
    control.orientation.y = orientation[1]
    control.orientation.z = orientation[2]
    control.orientation.w = orientation[3]
    control.interaction_mode = ims.InteractiveMarkerControl.MOVE_AXIS
    return control

def make_directional_controls(name, x=True, y=True, z=True):
    l = []
    
    if x:
        x_control = make_control_marker()
        x_control.orientation.x = 1
        x_control.name = name + "_move_x"
        l.append(x_control)
    
    if y:
        y_control = make_control_marker()
        y_control.orientation.y = 1
        y_control.name = name + "_move_y"
        l.append(y_control)

    if z:
        z_control = make_control_marker()
        z_control.orientation.z = 1
        z_control.name = name + "_move_z"
        l.append(z_control)

    return l

def make_orientation_controls(name, x=True, y=True, z=True):
    controls = make_directional_controls(name + '_rotate', x, y, z)
    for c in controls:
        c.interaction_mode = ims.InteractiveMarkerControl.ROTATE_AXIS
    return controls

