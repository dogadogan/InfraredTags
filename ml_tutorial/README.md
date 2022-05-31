# Overview

<p align="center">
    <img  src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/workflow.png">
</p>

The goal of the ML modules is to turn a low-resolution IR image to a binary image where the code is detectable. To achieve the goal, we have implemented the following workflow:

1. [Data Generation](https://github.com/dogadogan/InfraredTags/blob/main/ml_tutorial/data_generation.md)

    Generate images from videos. The images are split into test, val, and train.


2. [Data Augmentation](https://github.com/dogadogan/InfraredTags/blob/main/ml_tutorial/data_augmentation.md)

    Augment training data to increase the number of training images.


3. [Train](https://github.com/dogadogan/InfraredTags/blob/main/ml_tutorial/train.md)

    Train the Unet model with the augmented data.


4. [Deploy](https://github.com/dogadogan/InfraredTags/blob/main/ml_tutorial/deploy.md)

    Deploy the trained Unet model.



