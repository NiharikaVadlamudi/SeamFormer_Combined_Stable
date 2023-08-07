'''
Train File for scribble 
and binary branch training.
'''

# Library Imports 
import sys
import os
import copy
import torch
import wandb
import numpy as np
from vit_pytorch import ViT
import torch.optim as optim
from einops import rearrange

# File Imports 
from dataloader import *
import utils as utils
from network import SeamFormer
from configuration import config as settings    
from netutils import imvisualize

# Global Settings 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_of_gpus = torch.cuda.device_count()

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
    if settings['pretrained_weights_path'] is not None:
        try:
            network.load_state_dict(torch.load(settings['pretrained_weights_path'], map_location=device),strict=False)
            print('Pre-trained network loaded successfully!')
        except Exception as exp :
            print('Network Pre-Trained Weights Error: %s' % exp)
            sys.exit()

    # Freezing the model based on scribble/binarisation training.
    if settings['train_binary']:
        # Freezing every trainable parameter with scr 
        for name, param in network.named_parameters():
            if param.requires_grad and name.find("scr")>=0 :
                param.requires_grad = False

    elif settings['train_scribble']:
         for name, param in network.named_parameters():
            if param.requires_grad and not name.find("scr")>=0 :
                param.requires_grad = False
    else:
        print('Neither scribble_train nor binary_train , Exiting !')
        sys.exit()

    network = network.to(device)
    return network



# Learning Rate of Optimiser 
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

# Validation framework 
def validateNetwork(epoch,network,settings,validloader,vis=True):
    batch_size = np.int32(settings['batch_size'])
    patch_size = np.int32(settings['patchsize'])
    image_size = np.int32(settings['imgsize'])
    losses = 0
    network.eval()
    for bindex, (valid_index, valid_in, valid_out) in enumerate(validloader):
        try:
            inputs = valid_in.to(device)
            outputs = valid_out.to(device)
            with torch.no_grad():
                # --- Weight Calculation for BCE Loss ---# 
                outputs = outputs[:, 0, :, :].unsqueeze(1)
                n_black = len(outputs[outputs==0.0])
                n_white = len(outputs[outputs==1.0]) + 1
                weight = torch.tensor(n_black/n_white)
                loss_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')
                # Forward Pass 
                if settings['strain']:
                    loss,gt_patches,pred_pixel_values = network(inputs,gt_bin_img=None,gt_scr_img=outputs,criterion=loss_criterion,strain=True,btrain=False,mode='train')
                # Forward Pass - strain 
                if settings['btrain']:
                    loss,gt__patches,pred_pixel_values= network(inputs,gt_bin_img=outputs,gt_scr_img=None,criterion=loss_criterion,strain=False,btrain=True,mode='train')
            
                rec_images = rearrange(pred_pixel_values, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',p1 = patch_size, p2 = patch_size, h=image_size//patch_size)
                
                # Visualisation 
                if bindex%1000==0 and vis is True:
                    imvisualize(settings,inputs.cpu(), outputs.cpu(),rec_images.cpu(),bindex,epoch)

                losses += loss.item()
        except Exception as e :
            print('ValidationNetwork Error : {}'.format(str(e)))
            continue
    validationLoss = losses / len(validloader)
    return validationLoss

# Train ( Binary / Scribble ) with minimum of 100 samples .
def trainNetwork(settings,min_samples=100):
    if settings['enabledWandb']:
        experimentName=settings['experiment_base']+settings['wid']
    else:
        experimentName=settings['experiment_base']
    
    # Network
    network=buildModel(settings)

    # Weights Folder 
    os.makedirs(settings['model_weights_path'],exist_ok=True)
    
    #Get dataloaders
    trainloader, validloader = all_data_loader(settings['batch_size'])
    print('TrainLoader Samples : {} TestLoader samples : {}'.format(len(trainloader),len(validloader)))

    try: 
        assert len(trainloader)>min_samples and len(validloader)>min_samples , "Insufficient Samples !"
    except Exception as exp :
        print('TrainStepError : Exiting ! Error in Sample Loading -{}'.format(exp))
        sys.exit(0)

    # Optimizer 
    optimizer = optim.AdamW(network.parameters(),lr=settings['learning_rate'],betas=(0.9, 0.95),eps=1e-08, weight_decay=0.005,amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    
    # Epoch-wise iteration 
    global best_loss
    global best_epoch 

    best_loss = 100 
    best_epoch = -1 

    # We split the samples into batchSize , so undo it for obtaining the total samples
    train_samples = len(trainloader)*settings['batchsize']

    # Iterating epochs...
    for epoch in range(0,settings['num_epochs']):
        running_loss=0.0
        network.train()
        iters=0
        print('Epoch-wise iteration : {}'.format(epoch))
        batch_size = settings['batch_size']
        for i,(train_index, train_in, train_out) in enumerate(trainloader):
            try:
                iters+=1
                inputs = train_in.to(device)
                outputs = train_out.to(device)
                optimizer.zero_grad()

                # --- Weight Calculation ---
                outputs = outputs[:, 0, :, :].unsqueeze(1)
                n_black = len(outputs[outputs==0.0])
                n_white = len(outputs[outputs==1.0]) + 1
                weight = torch.tensor(n_black/n_white)
                loss_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')

                # Forward Pass - btrain 
                if settings['strain']:
                    loss,_,_ = network(inputs,gt_bin_img=None,gt_scr_img=outputs,criterion=loss_criterion,strain=True,btrain=False,mode='train')
                # Forward Pass - strain 
                if settings['btrain']:
                    loss,_,_ = network(inputs,gt_bin_img=outputs,gt_scr_img=None,criterion=loss_criterion,strain=False,btrain=True,mode='train')
                
                # backward pass
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if iters%500==0:
                    print('Epoch : {} Step : {} Loss :{}'.format(epoch,iters,float(running_loss/iters)))

            except Exception as e:
                print('Train step error : {} , going past it !'.format(e))
                continue
        
        # Reduce the LR at the end of the epoch
        scheduler.step()
        lr_ = optimizer.param_groups[0]['lr']
        # Losses 
        trainLoss = running_loss/train_samples
        validationLoss = validateNetwork(epoch,network,settings,validloader,vis=True)
        print('Epoch : {} Train Loss : {} Validation Loss @ {} is {}'.format(str(epoch),str(trainLoss),str(validationLoss)))

        # Logging ! 
        if settings['enabledWandB']:
            wandb.log({'epoch':epoch,'num_batches':iters})
            wandb.log({'epoch':epoch,'train_loss':trainLoss})
            wandb.log({'epoch':epoch,'lr':get_lr(optimizer)})
            
        # Saving Model Weights ( Periodically & Best Model Weights )
        if epoch%(epoch['weight_logging_interval'])==0:
            print('Network Weights @ {} saved !!'.format(epoch))
            torch.save(network.state_dict(),os.path.join(settings['model_weights_path'],'network-{}-{}.pt'.format(settings['expName'],epoch)))
        else:
            if validationLoss<best_loss:
                print('Storing the best weight based on validation loss : {}'.format(epoch))
                best_loss=validationLoss
                best_epoch=epoch
                torch.save(network.state_dict(),os.path.join(settings['model_weights_path'],'BEST-MODEL-{}-{}.pt'.format(settings['expName'],epoch)))

    
if __name__ == "__main__":
    print ("Training Operation Invoked...")
    trainNetwork(settings)
    print('Training Completed ~')
