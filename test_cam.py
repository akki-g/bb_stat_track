# camera_test.py
import cv2
import time

def test_camera():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        fps_counter += 1
        if fps_counter % 30 == 0:
            elapsed = time.time() - start_time
            print(f"FPS: {fps_counter/elapsed:.2f}")
        
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()