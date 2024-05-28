import cv2
import numpy as np
from ultralytics import YOLO
import serial,time

# Load the YOLOv8 model
model = YOLO('yolov8n-face.pt')

# Initialize the webcam
cap = cv2.VideoCapture(0)

frame_width = 1920
frame_height = 1080

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to capture frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform face detection
    results = model(frame)

    # Process the results
    for result in results:
        
        boxes = result.boxes.cpu().numpy()  # Convert to numpy array
        if len(boxes) == 0:
            continue  # Skip if no detections

        for box in boxes: # there could be more than one detection
            print(np.shape(box.xyxy))
            x1, y1, x2, y2 = box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            print(x1, y1, x2, y2)
 
            # print("class", box.cls)
            # print("xyxy", box.xyxy)
            # print("conf", box.conf)
        
        #plot the center of the face
        cv2.circle(frame,(x1+(x2-x1)//2,y1+(y2-y1)//2),2,(0,255,0),2)
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow('Webcam Face Detection', frame)
    cv2.rectangle(frame,(frame_width//2-50,frame_height//2-50),
                 (frame_width//2+50,frame_height//2+50),
                  (255,255,255),3)
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

