#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:28:56 2021

@author: thomas_yang
"""

from configuration import Config
import numpy as np
import cv2

def show_traing_val_image(images, gt_heatmap, gt_joint, pred, training = False):

    name_dict = {False:['val_img', 'val_hms'], True:['train_img', 'train_hms']}
    
    min_show_figures = images.shape[0] if images.shape[0]%Config.batch_size == 0 else images.shape[0]%Config.batch_size       

    imgs = np.concatenate((images[0:min(min_show_figures,4),...,::-1].numpy()*255).astype(np.uint8), axis = 1)
    imgs = cv2.resize(imgs, (imgs.shape[1]//4, imgs.shape[0]//4))
    
    gt_pre_heatmap = \
    np.concatenate((np.concatenate((gt_heatmap[0:min(min_show_figures,4)]), axis = 1),
                    np.concatenate((pred[0:min(min_show_figures,4), :, :, 0:1].numpy()), axis = 1)), axis = 0)            
    
    gt_pre_joint = \
    np.concatenate((np.concatenate(np.max(gt_joint[0:min(min_show_figures,4)], axis=-1), axis = 1),
                    np.concatenate(np.max(pred[0:min(min_show_figures,4), :, :, 10:22], axis=-1), axis = 1)), axis = 0)            
    
    gt_pre_joint = np.expand_dims(gt_pre_joint, axis=2)
    hms = np.concatenate((gt_pre_heatmap, gt_pre_joint), axis = 1)
    
    cv2.imshow(name_dict[training][0], imgs)
    cv2.imshow(name_dict[training][1], hms)
    cv2.waitKey(1)