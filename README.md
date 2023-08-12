# SeamFormer

This repository contains the public implementation of SeamFormer, A High Precision Text Line Segmentation for Handwritten Document presented in the ICDAR 2023 paper. 

This repository is a one-click setup, where the pipeline automatically captures evaluation metrics and visual results immediately after inference is triggered. In addition, we support both offline and online logging, with the option to directly log data to the user's WandB account. 

## Table of contents
---
1. [Getting Started](#getting-started)
2. [Model Overview](#model)
3. [Model Inference](#model)
4. [Training](#model)
    - [Preparing Data](#preparing-the-data)
    - [Preparing Configs](#Preparing-the-configuration-files)
    - [Stage-1](#stage-1)
    - [Stage-2](#stage-2)
5. [Finetuning](#finetuning) 
6. [Contact](#contact)

## Getting Started
---
To make the code run, install the necessary libraries preferably using [conda](https://www.anaconda.com/) or else [pip](https://pip.pypa.io/en/stable/) environment manager.

```bash
conda create -n stage1 python=3.7.11
conda activate stage1
pip install -r stage1_requirements.txt
```

## Model 
---
Overall Two-stage Architecture: Stage-1 generated binarised output with just text content along with a scribble map. Stage-2 uses these two intermediate outputs to generate Seams and finally the required text-line segmentation. 

![Overall Architecture](readme_imgs/overall.png)

<br>
Stage - 1: Uses Encoder-Decoder based multi-task vision transformer to generate binarisation result in one branch and scribble(strike-through lines) in another branch.

<center> 

  ![stage 1](readme_imgs/stage1.png)  
</center>

<br>
Stage - 2: Uses binarisation and scribble output from previous stage to create custom energy map for Seam generation. Using which final text-line segments are produced

<br>

<center>

  ![stage 2](readme_imgs/stage2.png)  
</center>



## Model Inference
---

### Usage
- Possibly a colab file?? (something like in docentr)
- model zoo

### Output
- Intermediate results - Scribbles
- Final output samples


## Training
---
The SeamFormer is split into two parts:
- Stage-1: Binarisation and Scribble Generation
- Stage-2: Seam generation and final segmentation prediction

Download Pretrained weights for binarisation from [ drive link of DocEnTr: An End-to-End Document Image Enhancement Transformer]() and change the *pretrained_weights_path* in the json files in `configs` directory accordingly.

<br>

### Preparing the Data
To train the model dataset should be in a folder following the hierarchy: 

```
├── Dataset
│   ├── <Dataset>_Train
│   │   ├── imgs
│   │   ├── bin_imgs
│   │   ├── train.json
│   ├── <Dataset>_Test
│   │   ├── imgs
│   │   ├── bin_imgs
│   │   ├── test.json
│
├── ...
```

### Preparing the configuration files

`<dataset_name>_<exp_name>_Configuration.json`

- A table as in Doc-UFCN

  | Parameters  | Description | Default Value
  | ----------  | ----------- | ------------- |
  | dataset_code   | Short name for Dataset as in dataset folder   | I2 | 
  | experiment_base   |   | Content | 
  | wid   |    | Content | 
  | data_path   | Dataset path   | /ICDAR2023/I2 | 
  | model_weights_path   | Path location to store trained weights  | weights/ | 
  | visualisation_folder   |   | Content | 
  | learning_rate   | Initial learning rate of optimizer (scheduler applied) | 0.005-0.0009 | 
  | vit_model_size   |    | Content | 
  | img_size   | ViT input size    | Content | 
  | patch_size   | ViT patch size   | Content | 
  | encoder_freeze   | To freeze encoder for scribble generation   | Content | 
  | encoder_layers   | Number of encoder layers in stage-1 multi-task transformer   | Content | 
  | encoder_heads   | Number of heads in MHSA    | Content | 
  | encoder_dims   | Dimension of token in encoder   | Content | 
  | batch_size   | Batch size for training   | Content | 
  | num_epochs   | Total epochs for training   | Content | 
  | vis_results   |    | Content | 
  | pretrained_weights_path   | Path location for pretrained weights(either for scribble/binarisation)   | Content | 
  <!-- | WANDB_API_KEY   | Content   | Content | 
  | WANDB_ENTITY   | Content   | Content | 
  | WANDB_RESUME   | Content   | Content |  -->


### Stage-1
Stage 1 comprises of a multi-task tranformer for binarisation and scribble generation.

- Details for Binarisation alone flag (train, inference)
- Details for Scribble alone flag (train, inference)
- Details for both (only inference)

#### Patch Generation for Binarisation
```

```
#### Binarisation alone experiment from Scratch 
- Based on flag load pretrain weights from DocEnTr
```

```

---

#### Patch Generation for Scribbles
```

```
#### Scribble generation alone experiment from Scratch 
- Based on flag load pretrain weights from Binarisation stage
```

```

#### WandB Setup (Optional)
- Include in Config file. Make changes in train.py
- Those who need wandb, make the visualization local - text file (metrics), folder(images) 
```

```



### Stage-2
---

### Finetuning
---


## Contact 
For any suggestions/contributions to the repository , please contact : <br />
Niharika Vadlamudi - niharika11988@gmail.com / niharika.vadlamudi@research.iiit.ac.in
