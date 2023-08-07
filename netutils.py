'''
General Utility Files 
- Consisting of helper functions 
for supporting visualisations
of model results
'''

import os
import numpy as np
import math
import cv2

global mean , std 

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
    


def merge_images_horizontally(img1, img2, img3):
    assert img1.shape == img2.shape==img3.shape , "Error merging the images"
    # Resize images if necessary to make them the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, img1.shape[:2])
    if img1.shape != img3.shape:
        img3 = cv2.resize(img3, img1.shape[:2])
    # Combine the images horizontally
    merged_img = np.hstack((img1, img2, img3))
    return merged_img



def imvisualize(settings,imdeg, imgt, impred, ind, epoch=0,threshold=0.4):
    """
    Visualize the predicted images along with the degraded and clean gt ones
    Args:
        imdeg (tensor): degraded image
        imgt (tensor): gt clean image
        impred (tensor): prediced cleaned image
        ind (str): index of images (name)
        epoch (str): current epoch
        setting (str): experiment name
    """

    # unnormalize data
    imdeg = imdeg.numpy()
    imgt = imgt.numpy()
    impred = impred.numpy()

    impred = np.squeeze(impred, axis=0)
    imgt = np.squeeze(imgt, axis=0)
    imdeg = np.squeeze(imdeg, axis=0)
    
    imdeg = np.transpose(imdeg, (1, 2, 0))
    imgt = np.transpose(imgt, (1, 2, 0))
    impred = np.transpose(impred, (1, 2, 0))

    # Only for the input image 
    for ch in range(3):
        imdeg[:,:,ch] = (imdeg[:,:,ch] *std[ch]) + mean[ch]
        
    # avoid taking values of pixels outside [0, 1]
    impred[np.where(impred>1.0)] = 1
    impred[np.where(impred<0.0)] = 0

    # thresholding now 
    # binarize the predicted image taking 0.5 as threshold
    impred = (impred>threshold)*1

    # Change to 0-255 range 
    imdeg=imdeg*255
    imgt=imgt*255
    impred=impred*255
    impred= impred.astype(np.uint8)

    # save images
    if not settings['enabledWandb']:
        base_dir = os.path.join(settings['visualisation_folder'],'epoch_{}'.format(epoch))
        epoch=str(epoch)
        os.makedirs(base_dir,exist_ok=True)
        out = merge_images_horizontally(imdeg[:,:,0],imgt,impred)
        cv2.imwrite(os.path.join(base_dir,str(ind)+'_combined.png'),out)

    return imdeg,imgt,impred

