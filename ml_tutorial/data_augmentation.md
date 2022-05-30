# Dataset Augmentation Tutorial

## Directory Structure

```
InfraTags
└── ml                                   
    └── augmentation.py
```
## Augmentation

The code augment the generated data with following techniques:
- translation
- horizontal flip
- vertical flip
- rotation
- brightness adjustment
- contrast adjustment
- hue adjustment
- color jitter (include brightness, contrast and hue adjustments)

Note that the training set is augmented, but the validating and testing sets are not.

## Tutorial

#### 1) Change the name of the saving directory `save_dir`.

```python
save_dir = 'temp/'
Path(save_dir + 'test/input').mkdir(parents=True, exist_ok=False)
Path(save_dir + 'test/output').mkdir(parents=True, exist_ok=False)
Path(save_dir + 'train/input').mkdir(parents=True, exist_ok=False)
Path(save_dir + 'train/output').mkdir(parents=True, exist_ok=False)
Path(save_dir + 'val/input').mkdir(parents=True, exist_ok=False)
Path(save_dir + 'val/output').mkdir(parents=True, exist_ok=False)
```

#### 2) List all paths to `dataset_images_ml`, all table files `image_dataset.csv`, and types. All `dataset_images_ml` and `image_dataset.csv` are generated during data generation process. The order of elements in the three lists must correspond to the same generated data. For ArUco, type is the ID of ArUco code in the corresponding images and table files. For QR codes, type is the number we labeled for QR codes in our case. Below is the example.

```python
paths = [\
    "../opencv/datasets/dataset_AR1_bf0/dataset_images_ml",\
    "../opencv/datasets/dataset_AR1_bn0/dataset_images_ml",\
    "../opencv/datasets/dataset_AR1_wnf0/dataset_images_ml",\
    "../opencv/datasets/dataset_ARwheel_bf0/dataset_images_ml"
]

table_files = [\
    "../opencv/datasets/dataset_AR1_bf0/image_dataset.csv",\
    "../opencv/datasets/dataset_AR1_bn0/image_dataset.csv",\
    "../opencv/datasets/dataset_AR1_wnf0/image_dataset.csv",\
    "../opencv/datasets/dataset_ARwheel_bf0/image_dataset.csv"
]

types = [1,1,1,3]
```

#### 3) Include the following three lines. If you are working on ArUco, put `aruco=True`; otherwise put `aruco=False`.

```python
save_augmentation_all(paths, table_files, types, save_dir, 'test', aruco=True)
save_augmentation_all(paths, table_files, types, save_dir, 'train', aruco=True)
save_augmentation_all(paths, table_files, types, save_dir, 'val', aruco=True)
```

#### 4) Run `python3 augmentation.py` 


