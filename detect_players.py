# detect_players.py
import cv2
import numpy as np
from ultralytics import YOLO
import time

class PlayerDetector:
    def __init__(self):
        # Load YOLOv8n model
        self.model = YOLO('yolov8n.pt')
        
        # Basketball-relevant classes
        self.relevant_classes = {
            0: 'person',
            32: 'sports ball'
        }
        
    def detect_frame(self, frame):
        # Run inference
        results = self.model(frame, conf=0.5, classes=[0, 32])
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': self.relevant_classes.get(cls, 'unknown'),
                        'class_id': cls
                    })
        
        return detections
    
    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Different colors for person vs ball
            color = (0, 255, 0) if det['class'] == 'person' else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

# Test the detector
if __name__ == "__main__":
    detector = PlayerDetector()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect_frame(frame)
        frame = detector.draw_detections(frame, detections)
        
        cv2.imshow('Player Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()