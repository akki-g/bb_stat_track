import cv2
import numpy as np
import pickle

class CourtCalibrator:
    def __init__(self):
        # Standard basketball court dimensions (in meters)
        self.court_width = 15.0  # FIBA standard
        self.court_length = 28.0
        
        # Court corners in real-world coordinates
        self.real_corners = np.array([
            [0, 0],                    # Top-left
            [self.court_width, 0],     # Top-right
            [self.court_width, self.court_length],  # Bottom-right
            [0, self.court_length]     # Bottom-left
        ], dtype=np.float32)
        
        self.image_corners = None
        self.homography_matrix = None
        
    def calibrate_from_clicks(self, image_path):
        """Manual calibration by clicking court corners"""
        self.image = cv2.imread(image_path)
        self.clicked_points = []
        
        cv2.namedWindow('Court Calibration')
        cv2.setMouseCallback('Court Calibration', self._mouse_callback)
        
        print("Click on court corners in order: top-left, top-right, bottom-right, bottom-left")
        
        while len(self.clicked_points) < 4:
            display = self.image.copy()
            
            # Draw clicked points
            for i, pt in enumerate(self.clicked_points):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (pt[0]+10, pt[1]+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Court Calibration', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
        if len(self.clicked_points) == 4:
            self.image_corners = np.array(self.clicked_points, dtype=np.float32)
            self._calculate_homography()
            self._save_calibration()
            return True
        return False
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicked_points) < 4:
            self.clicked_points.append((x, y))
    
    def _calculate_homography(self):
        """Calculate homography matrix for perspective transformation"""
        self.homography_matrix = cv2.getPerspectiveTransform(
            self.image_corners, self.real_corners
        )
    
    def pixel_to_court(self, pixel_x, pixel_y):
        """Convert pixel coordinates to court coordinates"""
        if self.homography_matrix is None:
            return None
        
        point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(
            point.reshape(-1, 1, 2), self.homography_matrix
        )
        return transformed[0][0]
    
    def _save_calibration(self):
        """Save calibration data"""
        calibration_data = {
            'image_corners': self.image_corners,
            'real_corners': self.real_corners,
            'homography_matrix': self.homography_matrix
        }
        with open('court_calibration.pkl', 'wb') as f:
            pickle.dump(calibration_data, f)
    
    def load_calibration(self):
        """Load saved calibration"""
        try:
            with open('court_calibration.pkl', 'rb') as f:
                data = pickle.load(f)
                self.image_corners = data['image_corners']
                self.real_corners = data['real_corners']
                self.homography_matrix = data['homography_matrix']
                return True
        except:
            return False
        

        