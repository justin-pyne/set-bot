import cv2
import numpy as np

# Load the pre-trained YOLOv4 model
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')

# Load the card class names and colors
classes = []
with open('classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Open a connection to the camera
cap = cv2.VideoCapture(0)  # 0 for default camera, 1 for second camera, etc.

# Loop over frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the frame was not read correctly, exit the loop
    if not ret:
        break

    # Get the image dimensions
    height, width, channels = frame.shape

    # Create a blob from the image for input to the neural network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the neural network and forward pass to get the output detections
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    # Loop over the detected objects and filter for cards
    cards = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'card':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                card_width = int(detection[2] * width)
                card_height = int(detection[3] * height)
                left = int(center_x - card_width / 2)
                top = int(center_y - card_height / 2)
                cards.append((left, top, card_width, card_height))
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove duplicate detections
    indices = cv2.dnn.NMSBoxes(cards, confidences, 0.5, 0.4)

    # Draw bounding boxes around the detected cards
    for i in indices:
        i = i[0]
        left, top, width, height = cards[i]
        color = colors[class_ids[i]].tolist()
        cv2.rectangle(frame, (left, top), (left+width, top+height), color, thickness=2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Wait for a key press and exit if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

