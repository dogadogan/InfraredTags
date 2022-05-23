# InfraredTags Tutorial
   InfraredTags is a low cost way to detect unobtrusive tags. The project has been published at the 2022 ACM CHI Conference on Human Factors in Computing Systems. Learn more about the project [here](https://hcie.csail.mit.edu/research/infraredtags/infraredtags.html)

<p align="center">
  <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/InfraredTags_Teaser.png">
</p>

## #1 CAD: Adding the tag into the object
### Requirements
- Rhino 6 3D  (make sure its version 6) and [grasshopper](https://www.rhino3d.com/download/archive/rhino-for-windows/6/latest/)
        - Once installed follow the instructions to install the pufferfish [plugin](https://github.com/HCIELab/InfraTags/tree/main/public/encoder/plugins/Pufferfish3-0)
- Python and IDE (Any IDE will work, however we use [pycharm](https://www.jetbrains.com/pycharm/download/#section=windows))
### Using the Encoder
#### 1) Open the Encoder_V1.gh file:
<p align="center">
  <img height="150" src="readme_img/file.png">
</p>

#### 2) Import STL file:
<p align="center">
  <img  height="200" src="/readme_img/inputs_stl.png">
</p>

#### 3) Import SVG (Here is where it gets complicated slightly)
   -  the SVGs have to be in a very specific format in order for our Grasshopper code to parse it (see below) 

<p align="center">
  <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/inputs_svg.png">
</p>

- Once imported you should see this:

<p align="center">
  <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/output_sm.png">
</p>

#### 4) Change the x,y,z coordinate of the point to move the code around on the surface of the object.
   - Best way to move a point is to simply set the coordinates by right clicking "Pt" on the inputs panel and then going to manage collection and typing a new point
   - Due to a bug in the code its best to keep the point in the positive Z axis.


<p align="center">
  <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/input_pt.png">
</p>
<p align="center">
  <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/input_manage_collection.png">
</p>



#### 5) Set top layer thickness and air gap thickness (1.38 and 1.92 respectively for white pla and IR pla)

#### 6) Export STLs:
   - For Single Material right click "single material" and click bake. For Multi-Material right click on both "multi material" and "IR filament"
   - A black wire mesh should appear in the perspective screen
   - simply highlight it with your mouse then navigate to File > Export selected and save somewhere in your file system
   - Note: for multi material you need to bake and export each mesh seperatley that way you have both the internal pla component and the outer IR pla component. 
<p align="center">
   <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/sm_save.png">
</p>
<p align="center">
   <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/sm_mesh.png">
</p>
<p align="center">
   <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/code_mesh.png">
</p>

#### SVG Formating
   - To format the SVG you have two options:
      - Take original SVG and parse it into paths of the following format: ```<path d="Mx,yhavbh-az"></path>```
         - x,y are the starting position 
         - A is horizontal length, b is vertical length
         - Ex:  ```<path d="M0,0h5v6h-5z"></path>``` 
      - An easier solution is to use webisites that generate the codes automatically and some python :
         - For QR codes use svgs generated by this library (https://www.nayuki.io/page/qr-code-generator-library).
         - For Aruco get svgs from this library (https://chev.me/arucogen/) and then save the svg and keep track of its path then pass it into the "Aruco_to_Path.py" file changing the paths in line 84 and 85. 

## #2 Fabrication: 3D printing the object
#### Materials
   - Although our technique can be used with many filaments we recommend using white PLA and IR PLA ([link to IR pla](https://3dk.berlin/en/special/115-pla-filament-ir-black.html))
  
#### 1) Open Cura Slicer or any Slicer that supports multi-material prints 

#### 2) import both models (IR PLA and Regular PLA)

<p align="center">
   <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/slr_import.png">
</p>

#### 3) arrange and slice   
    
<p align="center">
   <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/slr_arrange.png">
</p>


<p align="center">
   <img  height="200" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/slr_gcode.png">
</p>

## #3 Detection: Reading the tags
#### Materials
 - Pi NIR Camera [(link)](https://www.amazon.com/kuman-Raspberry-Camera-Module-Supports/dp/B0759GYR51/ref=sr_1_25?crid=IJE0D6SB8AQG&keywords=PI+noir+5MP&qid=1653020486&s=electronics&sprefix=pi+noir+5mp%2Celectronics%2C88&sr=1-25)
 - Raspberry PI Zero and included products [(link)](https://www.raspberrypi.com/products/raspberry-pi-zero/)
 - Micro-USB to USB type A cable [(link)](https://www.amazon.com/AmazonBasics-Male-Micro-Cable-Black/dp/B07232M876/ref=sr_1_3?keywords=micro+usb+to+usb&qid=1653020580&sr=8-3)
 - (Optional) Camera Case to house all parts [(link)](https://github.com/HCIELab/InfraTags/tree/main/public/hardware/Camera_Case)
#### Hardware 
   - Once a raspberry pi and NIR camera are obtained. follow the instructions at this [link](https://tutorial.cytron.io/2020/12/29/raspberry-pi-zero-usb-webcam/) and follow the instructions to set up the pi + camera as a usb camera 
#### Software
   - It is recommend that you use pycharm to run the decoder demos both for QR and Aruco, however the code can be run from a terminal 
   - Have Python3 and pip3 pre-installed on your system link for this is [here](https://www.python.org/downloads/) version 3.6 or greater should work just fine
   - Run the following command in terminal:
      - ```pip install opencv-python numpy dbr opencv-contrib-python pyzbar```
   - Or in pycharm navigate to File > Settings > Project > Python Interpreter > Install packages (click the plus sign) and install the following packages:
      - opencv-python 
      - numpy
      - dbr
      - opencv-contrib-python
      - pyzbar
## Using the Decoder
### QR 
   - Navigate to qr_demo > qr_demo.py 
   - Open the file in an editor
   - Navigate to line 22 and confirm that CAMERA_STREAM is 1, indicating the usb camera
   - You should see a window popup in your screen if everything went alright 
   - There should also be a terminal outputting data on whether a code was detected or not
   - [put pictures here dont have pi IR camera to put images]
   
### Aruco
   - Navigate to aruco_demo > aruco_demo.py
   - Open the file in an editor
   - Navigate to line 20 and confirm that CAMERA_STREAM is 1, indicating the usb camera
   - You should see a window popup in your screen if everything went alright
   - There should also be a terminal outputting data on whether a code was detected or not
   - [put pictures here dont have pi IR camera to put images]
  
### Calibratiing the Image Transforms for Aruco Code
You should only do this if you want to change the Parameters for the Aroco detection
   - Navigate to infrared_python_api and open irtags_calib.py 
   - Navigate to line 17 and confirm VIDEO_STREAM is 1 for the usb IR camera
   - A window with a panel should open on the right play around with the values until a code is detected 
   - Take note of these values, these values can be used to change the parameters for the image transforms
  
### Oneplus
   #### hardware
   - OnePlus 8 Pro (found [here](https://www.oneplus.com/8-pro)) with Android [insert android version here]

   #### Software
   - ADB shell ([insallation guide](https://www.xda-developers.com/install-adb-windows-macos-linux/))
   - Follow the steps to enable wireless debugging on the OnePlus and pair with your PC ([here](https://medium.com/android-news/wireless-debugging-through-adb-in-android-using-wifi-965f7edd163a))
   
   #### Decoding
   - Once everything is installe, and you have paired the OnePlus to your computer via adb, you can run this command to get the IR camera show up:
   ```adb shell am start -n com.oneplus.factorymode/.camera.manualtest.CameraManualTest``` (more detail [here](https://www.xda-developers.com/oneplus-8-pro-color-filter-camera-still-accessible-adb-command/))
   - You should see the IR stream open on the OnePlus:
    <p align="center"> <img  height="450" src="https://github.com/HCIELab/InfraTags/blob/main/public/readme_img/oneplus_ir.png"> </p>
   - Note: if you do not see the IR camera, you may have to change the camera view to camera view 4 as seen in the top right of the image
   - It is important that once you are in camera view 4/IR camera view do not change views again. The app will freeze and you will need to restart the phone and resend the command to open the IR view again. 
   - Finally, after all this is done navigate to the oneplus folder ([here](https://github.com/HCIELab/InfraTags/tree/main/public/oneplus)) and run onplus.py. This should open up a window on your PC, livestreaming the oneplus' screen
  
### Additional Comments on DBR decoder
 - All the demo code above for detecting QR codes uses DBR in the backend. It comes with a 1-day trial license which must be renewed to get the messages from decoding. 
 - If you do not renew the license you will get only partial decoding of messages.
 - To update the license key navigate to the dbr_decode.py file for each demo and change the license key variable (line 4 of dbr_decode.py)
 