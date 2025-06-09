# shot_detection.py
import numpy as np
from collections import deque

class ShotDetector:
    def __init__(self, court_calibrator):
        self.court_calibrator = court_calibrator
        self.ball_trajectory = deque(maxlen=30)  # 1 second at 30fps
        self.shot_in_progress = False
        self.shot_start_frame = None
        
        # Basketball hoop position (approximate)
        self.hoop_positions = [
            (7.5, 1.575),   # One end
            (7.5, 26.425)   # Other end
        ]
        
    def update(self, detections, frame_number):
        """Process detections and identify shots"""
        ball_detections = [d for d in detections if d['class'] == 'sports ball']
        
        shot_detected = False
        
        if ball_detections:
            # Use the most confident ball detection
            ball = max(ball_detections, key=lambda x: x['confidence'])
            
            # Get ball center in pixels
            x1, y1, x2, y2 = ball['bbox']
            ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Convert to court coordinates
            court_pos = self.court_calibrator.pixel_to_court(
                ball_center[0], ball_center[1]
            )
            
            if court_pos is not None:
                self.ball_trajectory.append({
                    'frame': frame_number,
                    'pixel_pos': ball_center,
                    'court_pos': court_pos,
                    'bbox': ball['bbox']
                })
                
                # Analyze trajectory for shot detection
                shot_detected = self._analyze_trajectory()
        
        return shot_detected
    
    def _analyze_trajectory(self):
        """Analyze ball trajectory for shot patterns"""
        if len(self.ball_trajectory) < 10:
            return False
        
        # Get recent trajectory
        recent_points = list(self.ball_trajectory)[-10:]
        
        # Calculate vertical movement
        y_positions = [p['pixel_pos'][1] for p in recent_points]
        y_diff = y_positions[0] - y_positions[-1]  # Negative = moving down
        
        # Detect upward then downward motion (parabolic trajectory)
        if not self.shot_in_progress and y_diff > 30:  # Ball moving up
            # Check if ball is near a player
            self.shot_in_progress = True
            self.shot_start_frame = recent_points[0]['frame']
            
        elif self.shot_in_progress and y_diff < -20:  # Ball coming down
            # Check if near hoop
            current_court_pos = recent_points[-1]['court_pos']
            
            for hoop_pos in self.hoop_positions:
                distance = np.sqrt(
                    (current_court_pos[0] - hoop_pos[0])**2 + 
                    (current_court_pos[1] - hoop_pos[1])**2
                )
                
                if distance < 3.0:  # Within 3 meters of hoop
                    self.shot_in_progress = False
                    return True
        
        # Reset if trajectory is too long without completion
        if (self.shot_in_progress and 
            len(self.ball_trajectory) > 0 and
            self.ball_trajectory[-1]['frame'] - self.shot_start_frame > 90):
            self.shot_in_progress = False
        
        return False
    


    