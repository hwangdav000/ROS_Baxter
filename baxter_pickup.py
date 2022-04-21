#!/usr/bin/python2
import os
import sys
import time
import rospy
import subprocess
import baxter_interface
from baxter_interface import CHECK_VERSION
from baxter_interface.camera import CameraController
import joint_position_file_playback
from joint_position_file_playback import map_file
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

bridge = CvBridge()

# Open camera(camera is a string and res is a2-elementvector)
def open_cam(camera, res):
# Check if valid resolution
	if not any((res[0] == r[0] and res[1] == r[1]) for r in CameraController.MODES):
		rospy.logerr("Invalid resolution provided.")
	# Open camera
	cam = CameraController(camera)  # Create camera object
	cam.resolution = res    # Set resolution
	cam.open()  # open

# Close camera
def close_cam(camera):
	cam = CameraController(camera)  # Create camera object
	cam.close() # close

# take image and save it
def image_callback(msg):
	print("Received an image!")
	try:
		cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
	except CvBridgeError, e:
		print(e)
	else:
		cv2.imwrite('test_image.png', cv2_img)

def main():
    rospy.loginfo("Initializing node... ")

    # enable robot
    rospy.init_node('rsdk_robot_start')
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    try:
        rs.enable()
    except Exception, e:
        rospy.logerr(e.strerror)
        return 0
    rospy.loginfo("Enable completed")

    # take snapshot
    # enable relevant cameras
    # Close left_hand_camera
    # close head camera in case already open
    close_cam('head_camera')
    close_cam('left_hand_camera')

	# # Open head_camera and set resolution to 320x200
    open_cam('head_camera',(320,200))

    # open up rqt
    subprocess.Popen("rqt_image_view")
    
    # get input to see when to take picture
    raw_input("Press Enter to capture image\n")

    # save snapshot
    image_topic = "/cameras/head_camera/image"
    sub = rospy.Subscriber(image_topic, Image, image_callback)
    rospy.wait_for_message(image_topic, Image)
    
    # stop subscribing
    sub.unregister()
    
    # classification for digit
    

    # From digit playback
    rospy.loginfo("Starting digit playback")
    digit = 3

    # check to see if robot is actually enabled
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()
    rospy.on_shutdown(clean_shutdown)

    # run joint playback on specified digit
    if (digit == 1):
        block1_file = "1_block.txt"
        map_file(block1_file)
    elif (digit == 2):
        block2_file = "2_block.txt"
        map_file(block2_file)
    elif (digit == 3):
        block3_file = "3_block.txt"
        map_file(block3_file)
    else:
        rospy.loginfo("Invalid digit")
        return 0
    rospy.loginfo("Digit playback finished")

    return 1
    
if __name__ == '__main__':
    sys.exit(main())