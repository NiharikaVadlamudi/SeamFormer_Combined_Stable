'''
General Utility Files 
- Consisting of helper functions 
for supporting visualisations
of model results
'''

import os
import math
import cv2
import copy 
import numpy as np
from empatches import EMPatches
from einops import rearrange

global mean , std 

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
    
# @WASEEM'S PNSR FOR FULL SCALE IMAGE WILL BE ADDED HERE 


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



def preprocess(deg_img):
    deg_img = (np.array(deg_img) /255).astype('float32')
    # normalize data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    out_deg_img = np.zeros([3, *deg_img.shape[:-1]])
    for i in range(3):
        out_deg_img[i] = (deg_img[:,:,i] - mean[i]) / std[i]
    return out_deg_img

def readFullImage(path,PDIM=256,DIM=256,OVERLAP=0.25):
    input_patches=[]
    emp = EMPatches()
    try:
        img = cv2.imread(path)
        img = preprocess(img)
        img = np.transpose(img)
        img_patches, indices = emp.extract_patches(img,patchsize=PDIM,overlap=OVERLAP)
        for i,patch in enumerate(img_patches):
              resized=[DIM,DIM]
              if patch.shape[0]!= DIM or patch.shape[1]!= DIM :
                  resized=[patch.shape[0],patch.shape[1]]
                  patch = cv2.resize(patch,(DIM,DIM),interpolation = cv2.INTER_AREA)
              # cv2_imshow(patch)
              patch = np.asarray(patch,dtype=np.float32)
              patch =  np.transpose(patch)
              patch= np.expand_dims(patch,axis=0)
              sample={'img':patch,'resized':resized}
              input_patches.append(sample)
    except Exception as exp :
        print('ImageReading Error ! :{}'.format(exp))
        return None,None
    return input_patches,indices


'''
Reconstruct from pred_pixels to patches 
'''
def reconstruct(pred_pixel_values,patch_size,target_shape,image_size):
    rec_patches = copy.deepcopy(pred_pixel_values)
    output_image = rearrange(rec_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',p1 = patch_size, p2 = patch_size,  h=image_size[0]//patch_size)
    output_image = output_image.cpu().numpy().squeeze()
    output_image =  output_image.T
    # Resizing to get desired output
    output_image = cv2.resize(output_image,target_shape, interpolation = cv2.INTER_AREA)
    # Basic Thresholding
    output_image[np.where( output_image>1)] = 1
    output_image[np.where( output_image<0)] = 0
    return output_image




