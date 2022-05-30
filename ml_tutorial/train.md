# ML Train Module Tutorial

## Kaggle Notebook

[Kaggle Notebook](https://www.kaggle.com/code/veerapatr/train-dbr-default/notebook)

## Tutorial

#### 1) Create Kaggle account, and ask for permission to access the notebook. (Send an email to veerapatr.yot@gmail.com?)

#### 2) Fork the notebook by clicking the Copy and Edit button. The Edit button in the below image should be replaced by the Copy and Edit button in your view.
<p align="center">
    <img  src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/fork.png">
</p>

#### 3) If you want to add data, you can click the Add data at the top right corner. Then click upload, select the zip file of your data, and name the dataset. You can also choose whether you want the data to be private or public. Last, click Create.

<p align="center">
    <img  height="150" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/add.png">
</p>

<p align="center">
    <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/upload.png">
</p>

<p align="center">
    <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/name.png">
</p>

#### 4) Before you run the notebook, make sure that the settings are correct. The language is Python, and you must select GPU for accelerator. Also, check if the desired dataset is loaded into the notebook by checking on the Input on the right hand side.
<p align="center">
    <img  height=  "200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/setting.png">
</p>

#### 5) In case you run the code elsewhere, there might be a problem when dealing with ArUCo. The normal opencv library does not support ArUCo. You should uninstall opencv-python and install opencv-contrib-python instead. The notebook on Kaggle has already dealed with this problem, and you can skip this step.

#### 6) UNet. The binarization model that we use is UNet. The architecture is as follows. You only need to specify in_channels and out_channels when initiating a model. Still, keep in mind that the size of the training image must be divisible by 16 because UNet performs maxpooling 4 times.

<p align="center">
    <img  height=  "300" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/unet_simple.png">
</p>

#### 8) Below is the code for training the model.
```python
# Check if you are using GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using the GPU!")
else:
    print("WARNING: Could not find GPU! Using CPU only")

#Bianarization
input_size = 96
save_dir = "/kaggle/working/weights_binarization"
save_all_epochs = True
batch_size = 16
lr = 0.001
num_epochs = 50
root_dir = "/kaggle/input/ar-wheel-96/dataset_ARwheel_96"
dataloader = get_dataloader_binarization(input_size, batch_size, root_dir)
criterion = nn.BCEWithLogitsLoss()
in_channels = 1 # rgb
out_channels = 1 # binary
model = Unet(in_channels, out_channels)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
aruco_flag = True

train_binarization(model, dataloader, criterion, optimizer, save_dir, save_all_epochs, num_epochs, aruco_flag)
```
- input_size: Your training image size is input_size x input_size.
- save_dir: The directory for saving the model
- save_all_epochs: boolean, save models for all epochs or not
- batch_size: batch_size
- lr: learning rate
- num_epochs: number of epochs
- root_dir: root directory of data. 
- dataloader: dataloader object
- criterion: loss function. Should be nn.BCEWithLogitsLoss().
- in_channels: 3 if RGB. 1 if gray.
- out_channels: 1 because our output is binary image.
- aruco_flag: True if the model is for ArUCo; otherwise, False.

