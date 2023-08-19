## Training Setup for Sundanese Dataset

### Sample images
![Sample sundanese 1](../readme_imgs/CB-3-18-90-7.jpg)

![Sample sundanese 2](../readme_imgs/CB-3-18-90-12.jpg)

### Dataset
```
pip install gdown
gdown 1rMmjE4wyKnObMSNXQtHhuTlvcgWCzt93
mkdir data/
mv SD.zip data/
unzip data/SD.zip -d data/SD/ 
rm -rf data/SD/__MACOSX 
rm -rf data/SD.zip
```

or 

Download directly from [here](https://drive.google.com/file/d/1rMmjE4wyKnObMSNXQtHhuTlvcgWCzt93/view?usp=drive_link). Follow the below data hierarchy after unzipping it.

```
data
├── SD
│   ├── SD_TRAIN
│   │   ├── binarise
│   │   │   ├── imgs
│   │   │   ├── bin_imgs
│   │   ├── scribble
│   │   │   ├── imgs
│   │   │   ├── SD_TRAIN.json
│   ├── SD_TEST
│   │   ├── binarise
│   │   │   ├── imgs
│   │   │   ├── bin_imgs
│   │   ├── scribble
│   │   │   ├── imgs
│   │   │   ├── SD_TEST.json

```

### Experiment Json Configuration
- Refer: [SD_exp1_Configuration.json](../SD_exp1_Configuration.json)

### Data Preparation for training
- Train Data
```bash
python datapreparation.py \
 --datafolder '/data/' \
 --outputfolderPath '/SD_train_patches' \
 --inputjsonPath '/data/SD/SD_TRAIN/scribble/SD_TRAIN.json' \
 --binaryFolderPath '/data/SD/SD_TRAIN/binarise/bin_imgs'
```
- Validation/Test Data
```bash
python datapreparation.py \
 --datafolder '/data/' \
 --outputfolderPath '/SD_test_patches' \
 --inputjsonPath '/data/SD/SD_TEST/scribble/SD_TEST.json' \
 --binaryFolderPath 'data/SD/SD_TEST/binarise/bin_imgs'
```

### Training Binarisation Branch
```bash
python train.py --exp_json_path 'SD_exp1_Configuration.json' --mode 'train' --train_binary
```

### Training Scribble Branch
```bash
python train.py --exp_json_path 'SD_exp1_Configuration.json' --mode 'train' --train_scribble
```
