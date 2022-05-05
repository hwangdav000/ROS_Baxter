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
import torch
import torchvision.models as models
import resnet9_utils
from resnet9_utils import ResNet9

bridge = CvBridge()

# Open camera (opens desired camera in Baxter at tuple resolution)
def open_cam(camera, resolution):
    # Check if resolution exists
    exists = False
    for r in CameraController.MODES:
	if(resolution[0] == r[0] and resolution[1] == r[1]):
	    exists = True
    if(not exists):
	rospy.logerr("Input resolution does not exist")

    # Open camera
    cam = CameraController(camera)  # Creates camera object
    cam.resolution = resolution    # Sets resolution
    cam.open()

# Close camera
def close_cam(camera):
    cam = CameraController(camera)  # Creates camera object
    cam.close()

# Takes in ros_img message and writes it as an image to root directory
def convert_image(ros_img):
    print("ROS image received")
    # Creates cv image object using captured photo
    try:
	cv2_img = bridge.imgmsg_to_cv2(ros_img, "bgr8")
    # If image object cannot be created throws error
    except CvBridgeError, e:
	print(e)
    # Writes image object as png/jpg in root directory
    else:
	cv2.imwrite('number_image.png', cv2_img)

def load_mnist():
    mnist = datasets.MNIST(
        root='data',
        train='true',
        transform=ToTensor(),
        download=True,
    )
    print("*******",mnist)
    data_full = []
    for i in range(0,len(mnist)):
        if mnist[i][1] < 4 and mnist[i][1] != 0:
            data_full.append(mnist[i])
    
    """data_full = [data_full[key] for (key, label) in enumerate(data_full) if label == 1 or label == 2 or label == 3] """
    print("**",len(data_full))
    # Split dataset to train,test and validation

    # train, test and validation seperation
    train_data, test_data, valid_data = torch.utils.data.random_split(data_full,
                                                                      [13000, 3000, 2831])
    # how to visualize the data
    """ plt.imshow(data_full.data[0])
    plt.imshow(data_full.data[0], cmap='gray')
    plt.imshow(data_full.data[44], cmap='gray')
    # access the data but you need export it first
    data1 = data_full.data[20]
    plt.imshow(data1, cmap='gray')
    # how to visualize multiple images
    figure = plt.figure(figsize=(10, 10))
    cols, row = 5, 5
    for i in range(1, cols * row + 1):
        idx = torch.randint(len(data_full), size=(1,)).item()

        img, label = data_full[idx]

        figure.add_subplot(row, cols, i)
        plt.title('Number: ' + str(label))
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap='gray')
    plt.show()"""
    # let's put into a dict
    batch_size = 100
    loaders = {'train': torch.utils.data.DataLoader(train_data,
                                                    batch_size=batch_size,
                                                    shuffle=True),

               'test': torch.utils.data.DataLoader(test_data,
                                                   batch_size=batch_size,
                                                   shuffle=True),
               'valid': torch.utils.data.DataLoader(valid_data,
                                                    batch_size=batch_size,
                                                    shuffle=True),

               }
    
    # visualize the dict
    train_part = loaders.get('train')
    data2 = train_part.dataset
    element1 = data2[0][0].squeeze()
    plt.imshow(element1, cmap='gray')
    return train_part,loaders

def run_cnn(runOrPredict,img_path,save_path):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if(runOrPredict=='load_model'):
        train_part,loaders= resnet9_utils.load_mnist()
        
        
        device,model,criterion,learning_rate,optimizer,num_epochs,loss_list,loss_list_mean = resnet9_utils.model_setup()
        
        model = resnet9_utils.model_train_test_validate(save_path,model,num_epochs,train_part,loaders,device,optimizer,criterion,loss_list,loss_list_mean)
    elif(runOrPredict=='predict_model'):
        model = ResNet9(1,10)
        model = torch.load(save_path)
        #model.eval()
        
    ImgTensor = resnet9_utils.preprocess_custom_image(img_path)
    index = resnet9_utils.prediction_custom_image(ImgTensor,model)
    
    print("Predicted label for the custom image", index)

def main():
    rospy.loginfo("Initializing node... ")

    # Enable robot
    rospy.init_node('rsdk_robot_start')
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    try:
        rs.enable()
    except Exception, e:
        rospy.logerr(e.strerror)
        return 0
    rospy.loginfo("Enable completed")

    # Take snapshot
    # Enable relevant cameras
    # Close left_hand_camera
    # Close head camera in case already open
    close_cam('head_camera')
    close_cam('left_hand_camera')

    # Open head_camera and set resolution to 320x200 for image processing
    open_cam('head_camera',(320,200))

    # Open up rqt
    subprocess.Popen("rqt_image_view")
    
    # Get input to see when to take picture
    raw_input("Press Enter to capture image\n")

    # Save snapshot
    image_topic = "/cameras/head_camera/image"
    sub = rospy.Subscriber(image_topic, Image, convert_image)
    rospy.wait_for_message(image_topic, Image)
    
    # Stop subscribing
    sub.unregister()
    
    # Classification for digit
    #run_cnn("predict_model",'~/ros_ws/number_image.png','~/ros_ws/number_image.png')

    # From digit playback
    rospy.loginfo("Starting digit playback")
    digit = 3

    # Check to see if robot is actually enabled
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()
    rospy.on_shutdown(clean_shutdown)

    # Run joint playback on specified digit
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
