# Dataset Generation Tutorial

## Directory Structure

```
InfraTags
└── opencv                   
    ├── movies_2/        
    ├── datasets/              
    └── dataset_generator.py
```
- movies_2: This directory stores the videos for data generation.
- datasets: This directory stores generated data.

## Videos

[Videos link](https://drive.google.com/drive/folders/1aLmoxCttv6wybgwaA9BfPLBcW62sIFeg?usp=sharing)

## Tutorial
#### 1) Adjust scale in `data_generation.py`
This parameter scale down the resolution of the frame cut from the video. Usually, this parameter is fixed, but this depends on the size of the video, and your desired resolution. In the image below, you can change the number `2678` (scale down by multiply `0.2678`) to another number. 

<p align="center">
   <img src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/scale.png">
</p>

#### 2) Run `python3 data_generation.py -i filename -m mode -f fpr`
- filename: The path to the video that we want to generate data from
- mode: **dbr** for QR codes. **aruco** for ArUco. There are other modes for Qr codes. **pyzbar** is public but detects worse than dbr. **wechat** uses ML approach, and takes very long time.
- fpr: frame per read.

#### Generated data
```
datasets
└── dataset_ARwheel_wf0                  
    ├── dataset_images/    
    ├── dataset_images_ml
    │   ├──test
    │   │  ├──input
    │   │  ├──masked
    │   │  └──output
    │   ├──train
    │   │  ├──input
    │   │  ├──masked
    │   │  └──output
    │   └──val
    │      ├──input
    │      ├──masked
    │      └──output    
    └── image_dataset.csv
```

The generated data is inside the directory `datasets`. For example, dataset_ARwheel_wf0 is the generated data. `image_dataset.csv` store meta data of each image, which will be used in data augmentation. The data used for augmentation is inside `dataset_images_ml`. They are separated into `test`, `train`, and `val`. 