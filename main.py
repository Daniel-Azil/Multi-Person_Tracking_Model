from ultralytics import YOLO
import cv2
import cvzone
import math

# module used for tracking objects in the video
from sort import *



# pass in path to video file for detection operation
cap = cv2.VideoCapture("videos_and_images/people.mp4")



# retrieve the model weights for detection
model = YOLO("model_weights/yolov8n.pt")


# create a mask for the vidoe using cava.com. the mask region sets a image parameter to what region in
# the video should the objects be detected.

mask = cv2.imread("videos_and_images/mask_region.png") 


# tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # create instance of the SORT tracker

# create line on the road to count objects that passes through it

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]
 
totalCountUp = []
totalCountDown = []



# class name for detection

className = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
             'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
             'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
             'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
             'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
             'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
             'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
             'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
             'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
             'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]




while True:
    success, img = cap.read()

    # Resize the mask image to match the video frame dimensions
    # mask = cv2.resize(mask, (1280, 720))
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    mask_region = cv2.bitwise_and(img, mask)  # Apply the mask region to the video frame

    # read the image on the video frame diectly
    image_graphics = cv2.imread("videos_and_images/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, image_graphics, (750, 60))


    results = model(mask_region, stream=True) #it's recommended to set stream as True as it utilizes generators for efficiency.


    detections = np.empty((0, 5))
    # get the bounding box
    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            
            # Confidence
            # get prediction confidence/rate
            conf = math.ceil(box.conf[0]*100) / 100   # we do this operation "math.ceil(box.conf[0]*100) / 100" to get the confidence to 2 decimal places
            

            # Class Name detection

            # retieve the index of the class name from the model's prediction so we can pass it to the className list to retrieve the actual class name in string
            cls = int(box.cls[0])  # the int function is used in converting the original number being float to an integer 
            
            # display dtection of the only object we want to detect
            # retieve all object class name
            currentClass =  className[cls]

            # display detection if class is a people
            if (currentClass == "person") and conf > 0.49:
                # in include class name and prediction confidence into the bounding box
                # we use (max(0, x1), max(35, y1)) to ensure the text is not out of the image frame when objects are detected at the top of the frame
                # cvzone.putTextRect(img, f"{className[cls]} {conf}", (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
                
                # show bounding box for only classes specified 
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9)

                current_array = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, current_array))





    track_result = tracker.update(detections)

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
 


    for result in track_result:
        x1,y1,x2,y2,id = result 
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        print(result)
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f"ID: {int(id)}", (max(0, x1), max(35, y1)), scale=2, thickness=2, offset=10)



        # find the dot or center of each object
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)
        
        # check if the dot of each detected object has falls in the range to the red line to
        # count them and know the total people encountered
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
 
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
    
    cv2.putText(img,str(len(totalCountUp)),(930,155),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(img,str(len(totalCountDown)),(1200,155),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)



    cv2.imshow("Image", img)
    
    
    cv2.waitKey(1)  
