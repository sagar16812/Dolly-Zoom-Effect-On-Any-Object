import argparse
import datetime
from Frame import *
from Predict import *


count = cv2.cuda.getCudaEnabledDeviceCount()
print(count)

# construct the argument parse 
parser = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network ')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--generate", default = 'no', help="generate an output video")
parser.add_argument("--prototxt", default="graph.pbtxt",
                                  help='Path to text network file: '
                                       'graph.pbtxt'
                                       )
parser.add_argument("--weights", default="frozen_inference_graph.pb",
                                 help='Path to weights: '
                                      'frozen_inference_graph.pb'
                                      )
parser.add_argument("--thr", default=0.65, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--resize", default=300, type=float, help="value by which the frame will be resized")
args = parser.parse_args()

# Labels 
# classNames = ['background',
#             'person',
#             'bicycle',
#             'car',
#             'motorcycle',
#             'airplane',
#             'bus',
#             'train',
#             'truck'] 

##############################################################################################
#       SETTINGS        # What worked best for me:
##############################################################################################
ZOOM = 0.25             # Medium = 0.2 to 0.3,    Close = 0.35 to 0.5
SHOW_BOX = True         # Show object detection box around the largest detected object
##############################################################################################
##############################################################################################

with open('object_detection_classes_coco.txt') as f:
    classNames = f.read().splitlines()

# Open video file or capture device. 
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(-1)

# Initialize the network
detector = Detector(args.weights, args.prototxt)

avg_extime = []

fps = cap.get(cv2.CAP_PROP_FPS)
print("input video fps: {}".format(fps))

# Create global detection box for steady screen transformation
box = BoundingBox(-1, -1, -1, -1)


# Write the sample output video
video_size = (int(1280/2), int(720/2))
#video_size = (int(1280/2),720)

if (args.generate == "yes"):
    out = cv2.VideoWriter('video_out.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, video_size)

print("Press \'ESC\' to exit")
# Loop through each img
while True:
    start = datetime.datetime.now()
    # Capture img-by-img
    ret, img = cap.read()
    if img is None:
        print("End of the video")
        break
    # img_resized = cv2.resize(img,(300,300))
    rows, cols, channels = img.shape

    # Detect the object/objects in the image
    detections = detector.predict(img, int(args.resize))

    end = datetime.datetime.now()
    time_taken = (end - start).total_seconds() * 1000
    avg_extime.append(time_taken)
    # print("Time taken in prediction : {}".format(time_taken))
    # fps = 1000 / time_taken
    
    boxes = getBoxes(detections, args.thr, rows, cols)
      
    # Linear interpolate bounding box to dimensions of largest detected box
    if boxes.size > 0:
        boxLrg = largestBox(boxes)
        if box.dim[0] == -1:
            box = boxLrg
        else:
            box.lerpShape(boxLrg)
    
    # Print the selected bounding box
    # print(box)

    # Setup frame properties and perform filter
    frame = Frame(img, box)
    frame.boxIsVisible = SHOW_BOX
    frame.setZoom(ZOOM)
    frame.filter()
    box = frame.box

    # Display filtered image as an OpenCV window
    frame.show()

    # Stop if escape key is pressed
    # Press Space to show/not show the bounding box
    # Press Numpad-2 key to zoom out 
    # Press Numpad-8 key to zoom in 
    k = cv2.waitKey(30)
    if k == 27:   # Esc key
        break
    if k == 32:   # Space key
        SHOW_BOX = not SHOW_BOX
    if k == 50:    # Numpad 2
        ZOOM = max(ZOOM - 0.05, 0.01)
        print(ZOOM)
    if k == 56:    # Numpad 8
        ZOOM = min(ZOOM + 0.05, 0.99)
        print(ZOOM)

    # Save output Image
    out_img = frame.img
    if (args.generate == "yes"):
        out_img = cv2.resize(out_img, video_size)
        out.write(out_img)

# Calculate and print the average time taken to do the prediction
avg_extime_np = np.array(avg_extime)
print("average time: {}".format(np.average(avg_extime_np)))

# Release the VideoCapture object
cap.release()

# Release the output video object
if (args.generate == "yes"):
    out.release()
