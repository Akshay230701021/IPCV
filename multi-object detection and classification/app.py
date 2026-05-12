import cv2
import numpy as np
from flask import Flask, render_template, Response
import os

app = Flask(__name__)

# Path to model files
prototxt_path = os.path.join(os.path.dirname(__file__), "MobileNetSSD_deploy.prototxt")
model_path = os.path.join(os.path.dirname(__file__), "MobileNetSSD_deploy.caffemodel")

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def generate_frames():
    video_path = os.path.join(os.path.dirname(__file__), "test_video.mp4")
    camera = cv2.VideoCapture(video_path)
    
    if not camera.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        (h, w) = frame.shape[:2]
        
        # Prepare the frame for object detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        counts = {}
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.4:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                counts[label] = counts.get(label, 0) + 1
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                color = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                text = f"{label}: {confidence:.2f}"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw summary on frame
        y0, dy = 30, 25
        cv2.putText(frame, "DETECTIONS:", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, (cls, count) in enumerate(counts.items()):
            text = f"- {cls}: {count}"
            cv2.putText(frame, text, (15, y0 + (i + 1) * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)