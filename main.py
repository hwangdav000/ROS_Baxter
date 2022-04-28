#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:45:13 2022

@author: shilpadeshpande
"""

import resnet9_utils


def main():
    train_part,loaders= resnet9_utils.load_mnist()
    
    
    device,model,criterion,learning_rate,optimizer,num_epochs,loss_list,loss_list_mean = resnet9_utils.model_setup()
    
    model = resnet9_utils.model_train_test_validate(model,num_epochs,train_part,loaders,device,optimizer,criterion,loss_list,loss_list_mean)
    
    ImgTensor = resnet9_utils.preprocess_custom_image('/Users/shilpadeshpande/Downloads/test_1.png')
    
    index = resnet9_utils.prediction_custom_image(ImgTensor,model)
    
    print("Predicted label for the custom image", index)
    
    
if __name__ == "__main__":
    main()
    
    