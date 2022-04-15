#!/usr/bin/python2
import os
import sys
import time
import rospy
import baxter_interface
from baxter_interface import CHECK_VERSION
import joint_position_file_playback
from joint_position_file_playback import map_file

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

    # open head camera

    # save snapshot

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