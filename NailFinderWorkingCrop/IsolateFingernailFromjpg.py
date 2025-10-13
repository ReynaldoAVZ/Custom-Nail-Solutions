from PIL import Image
import cv2
import numpy as np
from PIL import Image
import librealsense2 as rs
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

index = 1

for i in range(30):

    # define file names:

    image_path = "./input_data/Frames/SingleHand_Box_UV_Color{0}.png".format(index) #defines path for .png needed for fingernail ID
    ColorFileName = 'output/Color/cropped_file_Color{0}.npy'.format(index) #defines name of the cropped color file
    DepthFileName = 'output/Depth/cropped_file_Depth{0}.npy'.format(index) #defines name of the cropped depth file
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Error: Unable to load image from path {image_path}")

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (if required)
    image = cv2.resize(image, (300, 300))  # Resize to match the model's input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # TensorFlow inference
    with tf.Graph().as_default() as model:
        graphDef = tf.GraphDef()
        with tf.gfile.GFile('./model/export_model_008/frozen_inference_graph.pb', "rb") as f:
            serializedGraph = f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name="")

        with tf.Session(graph=model) as sess:
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")

            # Feed the loaded image into the model
            (boxes, scores, labels, N) = sess.run(
                [boxesTensor, scoresTensor, classesTensor, numDetections],
                feed_dict={imageTensor: image}
            )

            # Process the output as usual
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)


            # Visualization
            output = cv2.imread(image_path)  # Reload original image for visualizations
            k = 0
            nail_boxes = []
            for (box, score, label) in zip(boxes, scores, labels):
                if score < 0.6:  # Adjust confidence threshold as needed
                    continue
                (startY, startX, endY, endX) = box
                startX = int(startX * output.shape[1])
                startY = int(startY * output.shape[0])
                endX = int(endX * output.shape[1])
                endY = int(endY * output.shape[0])

                NailBox = tuple([startX,startY,endY,endX])
                nail_boxes.append(NailBox)

                # Open the image
                image = Image.open(image_path)

                # Define the cropping box (left, upper, right, lower)
                crop_box = ((startX-15),(startY-15),(endX+15),(endY+15))

                # Load the .npy file
                dataRGB = np.load("./input_data/Frames/SingleHand_Box_UV_Color{0}.npy".format(index))
                dataDepth = np.load("./input_data/Frames/SingleHand_Box_UV_Depth{0}.npy".format(index))

                # Define cropping bounds
                min_x, max_x = startX-10, endX+10  
                min_y, max_y = startY-10, endY+10  

                # Crop the array using slicing
                cropped_dataRGB = dataRGB[min_y:max_y, min_x:max_x]
                cropped_dataDepth = dataDepth[min_y:max_y, min_x:max_x]

                # Save the cropped data (optional)
                np.save(ColorFileName, cropped_dataRGB)
                np.save(DepthFileName, cropped_dataDepth)


                # Crop the image
                #cropped_image = image.crop(crop_box)

                # Save the cropped image
                #cropped_image.save("cropped_image{0}.jpg".format(i))
                #cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)
                k += 1

                #img = Image.fromarray(cropped_dataRGB, 'RGB')
                #img.save("RGBCropped{0}.png".format(i))

                index +=1

            # Display the output
            #cv2.imshow("Output", output)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()


            
