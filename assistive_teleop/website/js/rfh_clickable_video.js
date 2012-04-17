var count_surf_wipe_right=count_surf_wipe_left=force_wipe_count=0;
var img_act = 'looking'
var norm_appr_left = norm_appr_right = driving = tool_state = false;
var MJPEG_QUALITY= '50'
var MJPEG_WIDTH = '640'
var MJPEG_HEIGHT = '480'

function camera_init(){
    //Image-Click Publishers
    node.publish('norm_approach_right', 'geometry_msgs/PoseStamped', json({}));
    node.publish('norm_approach_left', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_contact_approach_right', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_contact_approach_left', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_poke_right_point', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_poke_left_point', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_swipe_right_goals', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_swipe_left_goals', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_wipe_right_goals', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_wipe_left_goals', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_rg_right_goal', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_rg_left_goal', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_grasp_right_goal', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_grasp_left_goal', 'geometry_msgs/PoseStamped', json({}));
    node.publish('wt_surf_wipe_r_points', 'geometry_msgs/Point', json({}));
    node.publish('wt_surf_wipe_l_points', 'geometry_msgs/Point', json({}));
};


//function set_camera(cam) {document.getElementById('video').src='http://'+ROBOT+':8080/stream?topic=/'+cam+'?width=640?height=480?quality=10'};
function set_camera(cam) {
mjpeg_url = 'http://'+ROBOT+':8080/stream?topic=/'+cam+'?width='+MJPEG_WIDTH+'?height='+MJPEG_HEIGHT+'?quality='+MJPEG_QUALITY
document.getElementById('video').src=mjpeg_url
};


function click_position(e) {
	var posx = 0;
	var posy = 0;
	if (!e) var e = window.event;
	if (e.pageX || e.pageY) 	{
		posx = e.pageX;
		posy = e.pageY;
	}
	else if (e.clientX || e.clientY) 	{
		posx = e.clientX + document.body.scrollLeft
			+ document.documentElement.scrollLeft;
		posy = e.clientY + document.body.scrollTop
			+ document.documentElement.scrollTop;
	}	return [posx,posy]
};

function get_point(event){
	var point = click_position(event);
	click_x = point[0] - document.getElementById('video_container').offsetLeft 
	click_y = point[1] - document.getElementById('video_container').offsetTop 
	//log("Clicked on point (x,y) = ("+ click_x.toString() +","+ click_y.toString()+")");
	return [click_x, click_y]
};

function image_click(event){
	var im_pixel = get_point(event);
	if (window.img_act == 'surf_wipe') {
    surf_points_out = window.gm_point
    surf_points_out.x = im_pixel[0]
    surf_points_out.y = im_pixel[1]
    log('Surface Wipe');
        log('Surface Wipe '+window.arm().toUpperCase()+' '+window.count_surf_wipe_right.toString())
        node.publish('wt_surf_wipe_'+window.arm()+'_points', 'geometry_msgs/Point', json(surf_points_out));
        if (window.count_surf_wipe == 0){
           log("Sending start position for surface-aware wiping");
           window.count_surf_wipe = 1;
        } else if (window.count_surf_wipe == 1){
           log("Sending end position for surface-aware wiping");
           window.count_surf_wipe = 0;
           $('#img_act_select').val('looking');
           window.img_act = 'looking';
        }
    }else{
	get_im_3d(im_pixel[0],im_pixel[1])
	};
};

function get_im_3d(x,y){
	window.point_2d.pixel_u = x;
	window.point_2d.pixel_v = y;
    log('Sending Pixel_2_3d request, awaiting response');
	node.rosjs.callService('/pixel_2_3d',
                            '['+json(window.point_2d.pixel_u)+','+json(window.point_2d.pixel_v)+']',
                            function(msg){
                                log('pixel_2_3d response received');
                                point_3d=msg.pixel3d;
                                determine_p23d_response(point_3d)
                            })
};

function determine_p23d_response(point_3d){
    switch (window.img_act){
        case 'looking':
            log("Sending look to point command");
            window.head_pub = window.clearInterval(head_pub);
            pub_head_goal(point_3d.pose.position.x, point_3d.pose.position.y, point_3d.pose.position.z, point_3d.header.frame_id);
            break
        case 'head_nav_goal':
                log("Sending navigation seed position");
                node.publish('head_nav_goal', 'geometry_msgs/PoseStamped', json(point_3d));
                break
        case 'norm_approach':
            log('Sending '+window.arm().toUpperCase()+ ' Arm Normal Approach Command')
            node.publish('norm_approach_'+window.arm(), 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'grasp':
            log('Sending command to attempt to grasp object with '+window.arm().toUpperCase()+' arm')
            node.publish('wt_grasp_'+window.arm()+'_goal', 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'reactive_grasp':
            log('Sending command to grasp object with reactive grasping with '+window.arm().toUpperCase()+' arm');
            node.publish('wt_rg_'+window.arm()+'_goal', 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'wipe':
            node.publish('wt_wipe_'+window.arm()+'_goals', 'geometry_msgs/PoseStamped', json(point_3d));
            if (window.force_wipe_count == 0) {
                window.force_wipe_count = 1;
                log('Sending start position for force-sensitive wiping')
            } else if (window.force_wipe_count == 1) {
                window.force_wipe_count = 0;
                log('Sending end position for force-sensitive wiping')
            };    
            break
        case 'swipe':
            log('Sending command to swipe from start to finish with '+window.arm().toUpperCase+' arm')
            node.publish('wt_swipe_'+window.arm()+'_goals', 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'poke':
            log('Sending command to poke point with '+window.arm().toUpperCase()+' arm')
            node.publish('wt_poke_'+window.arm()+'_point', 'geometry_msgs/PoseStamped', json(point_3d));
            break
        case 'contact_approach':
            log('Sending command to approaching point by moving until contact with '+window.arm().toUpperCase()+' arm')
            node.publish('wt_contact_approach_'+window.arm(), 'geometry_msgs/PoseStamped', json(point_3d));
        };
        if (window.force_wipe_count == 0){
            $('#img_act_select').val('looking');
            window.img_act = 'looking';
        };
};
