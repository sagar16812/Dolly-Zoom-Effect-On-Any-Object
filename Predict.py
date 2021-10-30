import cv2
import numpy as np

with open('object_detection_classes_coco.txt') as f:
    classNames = f.read().splitlines()

class Detector:
    def __init__(self, weights, prototxt):
        # Load a model imported from Tensorflow
        self.net = cv2.dnn.readNetFromTensorflow(weights, prototxt)

    def predict(self, img, size):
        ## NOTE:MobileNet requires fixed dimensions for input image(s)
        ## so we have to ensure that it is resized to 300x300 pixels.
        ## set a scale factor to image because network the objects has differents size. 
        ## We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input; For Example-
        ##            blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)  # scalefactor=1.0 (no scaling)
        ## after executing this command our "blob" now has the shape:
        ## (1, 3, 300, 300)


        # NOTE: [blobFromImage] creates 4-dimensional blob from image.
        # Optionally (resizes and crops image from center, 
        # subtract mean values, scales values by scalefactor, 
        # swap Blue and Red channels.)
        
        # image : This is the input image we want to preprocess before passing it through our deep neural network for classification.
        # scalefactor : After we perform mean subtraction we can optionally scale our images by some factor. 
        #             This value defaults to `1.0` (i.e., no scaling) but we can supply another value as well. 
        #             It’s also important to note that scalefactor should be 1 / \sigma as we’re actually 
        #             multiplying the input channels (after mean subtraction) by scalefactor .
        # size : Here we supply the spatial size that the Convolutional Neural Network expects. 
        #     For most current state-of-the-art neural networks this is either 224×224, 227×227, or 299×299.
        # mean : These are our mean subtraction values. They can be a 3-tuple of the RGB means or they can be a 
        #     single value in which case the supplied value is subtracted from every channel of the image. 
        #     If you’re performing mean subtraction, ensure you supply the 3-tuple in `(R, G, B)` order, 
        #     especially when utilizing the default behavior of swapRB=True .
        # swapRB : OpenCV assumes images are in BGR channel order; however, the `mean` value assumes we are using RGB order. 
        #         To resolve this discrepancy we can swap the R and B channels in image by setting this value to `True`. 
        #         By default OpenCV performs this channel swapping for us.

        # blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), (127.5, 127.5, 127.5), False) # ssd Caffe Model
        # blob = cv2.dnn.blobFromImage(img, 0.007843, (int(args.resize), int(args.resize)), (127.5, 127.5, 127.5), swapRB=True, crop=False) # this is for ssd mobilenet version3
        # blob = cv2.dnn.blobFromImage(img, 1. / 255, (int(args.resize), int(args.resize)), (0,0,0), swapRB=True, crop=False) # this is for ssd mobilenet version3
        blob = cv2.dnn.blobFromImage(img, size=(size,size), swapRB=True, crop=False) # this is for ssd mobilenet version2
        # blob = cv2.dnn.blobFromImage(img, size=(int(args.resize), int(args.resize)), swapRB=True)

        #Set to network the input blob 
        self.net.setInput(blob)
        #Prediction
        detections = self.net.forward()
        return detections


def getBoxes(detections, thr, rows, cols):
    boxes = []
    # Loop on the outputs
    for detection in detections[0,0]:
        confidence = float(detection[2])
        class_id = int(detection[1])
        if class_id in [0,1,3,4,6,8,10] and confidence > thr:
            # print("Class ID: {}".format(class_id-1))
            label = classNames[class_id-1] + ": " + str(confidence)
            
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            boxes.append([left,top, right-left,bottom-top])
    boxes = np.array(boxes) 
    return boxes