#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:45:13 2022

@author: shilpadeshpande
"""
import torch
import torchvision.models as models
import resnet9_utils
from resnet9_utils import ResNet9



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
    
    
if __name__ == "__main__":
    # run_model or predict_model
    # image path
    # model save path
    
    run_cnn("predict_model",'/Users/shilpadeshpande/Downloads/1_shot.jpg','/Users/shilpadeshpande/Downloads/resnet.pth')
    
    