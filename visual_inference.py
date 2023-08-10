'''
Inference File : 
> Args to the file : Input Image Folder , Output Image Folder , Model Inference Weight Path 

> Remaining settings you can see.
> Helps you load the model for chosen model weight path 
> It visualises both binary and scribble outputs together 
> Stores the image results under the folder provided 

'''

import os 
import sys 
import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.vit import ViT
import argparse 

# Global settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# File Imports 
from netutils import *
from network import * 

# Argument Parser 
def addArgs():
    # Required to override these params
    parser = argparse.ArgumentParser(description="SeamFormer:Inference")
    parser.add_argument("--input_image_folder",type=str, help="Input Folder Path", required=True,default=None)
    parser.add_argument("--output_image_folder",type=str, help="Output Folder Path for storing bin & scr results",required=True,default=None)
    parser.add_argument("--model_weights_path",type=str,help="Model Checkpoint Weights",default=None)

    # Fixed Arguments ( override in special cases)
    parser.add_argument("--encoder_layers",type=int, help="Encoder Level Layers",default=6)
    parser.add_argument("--encoder_heads",type=int, help="Encoder Heads",default=8)
    parser.add_argument("--encoder_dims",type=int, help="Internal Encoder Dim",default=768)
    parser.add_argument("--img_size",type=int, help="Image Shape",default=256)
    parser.add_argument("--patch_size",type=int, help="Input Patch Shape",default=8)
    parser.add_argument("--split_size",type=int, help="Splitting Image Dim",default=256)
    parser.add_argument("--threshold",type=float,help="Prediction Thresholding",default=0.40)
    
    return vars(parser.parse_args()) 


'''
Takes in the default settings 
and args to create the network.
'''
# Network Configuration  
def buildModel(settings):
    # Encoder settings
    encoder_layers = settings['encoder_layers']
    encoder_heads = settings['encoder_heads']
    encoder_dim = settings['encoder_dims']
    patch_size = settings['patchsize']
    # Encoder
    v = ViT(
        image_size = settings['imgsize'],
        patch_size =  settings['patchsize'],
        num_classes = 1000,
        dim = encoder_dim,
        depth = encoder_layers,
        heads = encoder_heads,
        mlp_dim = 2048)
    
    # Full model
    network =  SeamFormer(encoder = v,
        decoder_dim = encoder_dim,      
        decoder_depth = encoder_layers,
        decoder_heads = encoder_heads,
        patch_size = patch_size)
    
    # Load pre-trained network + letting the encoder network also trained in the process.
    if settings['model_weights_path'] is not None:
        if os.path.exists(settings['model_weights_path']):
            try:
                network.load_state_dict(torch.load(settings['model_weights_path'], map_location=device),strict=True)
                print('Network Weights loaded successfully!')
            except Exception as exp :
                print('Network Weights Loading Error , Exiting !: %s' % exp)
                sys.exit()
        else:
            print('Network Weights File Not Found')
            sys.exit()

    network = network.to(device)
    network.eval()
    return network



'''
Performs both binary and scribble output generation.
'''
def imageInference(network,path,args,PDIM=256,DIM=256,OVERLAP=0.25,save=True):
    if not os.path.exists(path):
        print('Exiting! Invalid Image Path : {}'.format(path))
        sys.exit(0)
    else:
        weight = torch.tensor(1) #Dummy weight 
        parentImage=cv2.imread(path)
        emp = EMPatches()
        input_patches , indices = readFullImage(path,PDIM,DIM,OVERLAP)
        
        patch_size=args['patch_size']
        img_size = args['img_size']
        THRESHOLD = args['threshold']

        soutput_patches=[]
        boutput_patches=[]
        # Iterate through the resulting patches
        for i,sample in enumerate(input_patches):
            p = sample['img']
            target_shape = (sample['resized'][1],sample['resized'][0])
            with torch.no_grad():
                inputs =torch.from_numpy(p).to(device)
                # Pass through model
                loss_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')
                pred_pixel_values_bin,pred_pixel_values_scr=network(inputs,gt_bin_img=inputs,gt_scr_img=inputs,criterion=loss_criterion,strain=True,btrain=True,mode='test')

                # Send them to .cpu
                pred_pixel_values_bin = pred_pixel_values_bin.cpu()
                pred_pixel_values_scr = pred_pixel_values_scr.cpu()

                bpatch=reconstruct(pred_pixel_values_bin,patch_size,target_shape,image_size)
                spatch=reconstruct(pred_pixel_values_scr,patch_size,target_shape,image_size)

                # binarize the predicted image taking 0.5 as threshold
                bpatch = ( bpatch>THRESHOLD)*1
                spatch = ( spatch>THRESHOLD)*1

                # Append the net processed patch
                soutput_patches.append(255*spatch)
                boutput_patches.append(255*bpatch)

        try:
            assert len(boutput_patches)==len(soutput_patches)==len(input_patches)
        except Exception as exp:
            print('Error in patch processing outputs : Exiting!')
            sys.exit(0)
        
        # Restich the image
        soutput = emp.merge_patches(soutput_patches,indices,mode='avg')
        boutput = emp.merge_patches(boutput_patches,indices,mode='avg')

        # Transpose
        binaryOutput=np.transpose(boutput,(1,0))
        scribbleOutput=np.transpose(soutput,(1,0))
        
        return binaryOutput,scribbleOutput



'''
Performs Binary & Scribble 
Inference given imageFolder
'''
def Inference(args):
    # Get the model first 
    network = buildModel(args)
    # Check if the image folder is non-empty and not null 
    if not os.path.exists(args['input_image_folder']):
        print(f"Input folder '{args['input']}' does not exist.")
        sys.exit(0)
    else:
        file_names = os.listdir(args['input_image_folder'])
        folder_contents = [ os.path.join(args['input_image_folder'],f)  for f in file_names]
    
    # Make output directory if its not present 
    os.makedirs(args['output_image_folder'],exist_ok=True)
    # Make a seperate scribble & binary image folders 
    scr_folder = os.path.join(args['output_image_folder'],'scr')
    bin_folder =  os.path.join(args['output_image_folder'],'bin')
    os.makedirs(scr_folder,exist_ok=True)
    os.makedirs(bin_folder,exist_ok=True)

    for i,file_path in enumerate(folder_contents):
        try:
            bin,scr =  imageInference(network,file_path,args,PDIM=256,DIM=256,OVERLAP=0.25,save=True)
            cv2.imwrite(os.path.join(scr_folder,'scr_'+file_names[i]),scr)
            cv2.imwrite(os.path.join(bin_folder,'bin_'+file_names[i]),bin)
        except Exception as exp:
            print('Image :{} Error :{}'.format(file_names[i],exp))
            continue

    print('--- Completed Inference ---')
    

if __name__=='main':
    args = addArgs()
    Inference(args)
    