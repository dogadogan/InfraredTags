# ML Model Deployment Tutorial

## Tutorial

The code is in InfraTags/ml/demo_ml.py

Below is the example code for deploying QR code model and detailed explanation.

```python

# This might be different for each laptop.
cameraID = 1

def main2():
    cap = cv2.VideoCapture(cameraID)
    while cap.isOpened():
         start = time()

        # read frame
        ret, frame = cap.read()

        # resize to desired size. Here, 432x288
        height = 288
        width = int(frame.shape[1] * 288 / frame.shape[0])
        image = cv2.resize(frame, (width, height))

        # extract the middle 288x288 becasue our input is square image
        image = image[:, 112:400, :]

        # display window for input image
        cv2.imshow("Frame", image)
        cv2.moveWindow("Frame", 0, 0)
        cv2.resizeWindow("Frame", 288, 288)

        # cv2 read an image as BGR, so we need to convert it to RGB in case the
        # input is colored image. If the input is gray use cv2.COLOR_BGR2GRAY instead
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # turn image to tensor
        image_tensor = t(image)

        # Add the fourth dimension of batch becasue the model is trained as a batch.
        # The dimension is 1xCxWxH.
        image_tensor = torch.reshape(image_tensor, (1, image_tensor.size(dim=0), \
                        image_tensor.size(dim=1), image_tensor.size(dim=2)))

        if ret:
            # get output
            output_b = b_model(image_tensor)

            # We use sigmoid to make the output in the range of [0,1], and round those that's not integer.
            output_b = torch.round(sigmoid(output_b))

            # Get rid off the batch dimension, extract the image, and turn it back to Pillow Image object
            output_b = to_pil(output_b[0,:,:,:])

            # Turn the output to numpy array for DBR detector.
            output_b = np.array(output_b)

            # DBR detector needs 3 dimensions, so we need to stack the binary image.
            if len(output_b.shape) == 2:
                output_b = np.stack((output_b,)*3, axis=-1)
            
            if is_decodable(output_b, mode, detector):
                # Get text and coordinates of the code, then display it.
                decoded = detector.decode_buffer(output_b)
                text = decoded[0].barcode_text
                coors = decoded[0].localization_result.localization_points
                cv2.polylines(output_b, [np.array(coors)], True, (51,153,255), 2)
                draw_text(output_b, text, coors)
                print('Text:', text)
                end = time()
                print('Time:', end-start)
                print('-------------')

            # display output window for output image
            cv2.imshow("output", output_b)
            cv2.moveWindow("output", 0, 320)
            cv2.resizeWindow("output", 288, 288)

        if cv2.waitKey(10) & 0xFF == ord('q') :
            # break out of the while loop
            break

    cv2.destroyAllWindows()
    cap.release()

# Don't forget to initiate the necessary objects before calling the main function.
model_path = './weights_binarization/ar_trial_288/bestnet_30_decode.pt'
in_channels = 3 # rgb
out_channels = 1 # binary
b_model = Unet(in_channels, out_channels)

# If you run on this on your laptop without GPU, you need map_location=torch.device('cpu').
b_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
b_model.eval()
mode = 'dbr'
detector = get_dbr_detector()

t = transforms.ToTensor()
to_pil = transforms.ToPILImage()
sigmoid = nn.Sigmoid()

main2()
```
