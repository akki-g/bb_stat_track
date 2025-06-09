# main_app.py
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime

from detect_players import PlayerDetector
from court_calibration import CourtCalibrator
from player_tracking import SimpleTracker
from shot_detection import ShotDetector
from action_recognition import ActionRecognizer
from stats_engine import StatsEngine

class BasketballTracker:
    def __init__(self):
        # Initialize components
        self.detector = PlayerDetector()
        self.tracker = SimpleTracker()
        self.court_calibrator = CourtCalibrator()
        self.shot_detector = None  # Initialize after calibration
        self.action_recognizer = ActionRecognizer()
        self.stats_engine = StatsEngine()
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.processing_active = False
        
        # Performance monitoring
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
    def setup(self):
        """Initial setup and calibration"""
        print("Basketball Tracker Setup")
        
        # Load or perform court calibration
        if not self.court_calibrator.load_calibration():
            print("Court calibration required.")
            print("Please capture a frame with the full court visible...")
            
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('court_frame.jpg', frame)
                self.court_calibrator.calibrate_from_clicks('court_frame.jpg')
            cap.release()
        
        self.shot_detector = ShotDetector(self.court_calibrator)
        
        # Team setup
        print("\nTeam Setup:")
        print("Assign players by clicking on them when they appear.")
        print("Press 'a' for Team A, 'b' for Team B")
        
    def processing_thread(self):
        """Background thread for CV processing"""
        frame_number = 0
        
        while self.processing_active:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Object detection
                detections = self.detector.detect_frame(frame)
                
                # Update tracking
                tracked_players = self.tracker.update(detections)
                
                # Get ball detection
                ball_detections = [d for d in detections if d['class'] == 'sports ball']
                
                # Shot detection
                shot_detected = self.shot_detector.update(detections, frame_number)
                if shot_detected:
                    # Simple shot assignment - closest player
                    closest_player = self._find_closest_player_to_ball(
                        tracked_players, ball_detections
                    )
                    if closest_player:
                        self.stats_engine.record_shot(
                            closest_player['track_id'], 
                            True,  # Assume made for MVP
                            frame_number,
                            None
                        )
                
                # Action recognition
                actions = self.action_recognizer.update(
                    tracked_players, 
                    ball_detections[0] if ball_detections else None
                )
                
                for action in actions:
                    self.stats_engine.record_action(
                        action['track_id'],
                        action['action'],
                        frame_number
                    )
                
                # Package results
                result = {
                    'frame': frame,
                    'detections': detections,
                    'tracked_players': tracked_players,
                    'actions': actions,
                    'shot_detected': shot_detected
                }
                
                self.result_queue.put(result)
                frame_number += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def _find_closest_player_to_ball(self, players, ball_detections):
        """Find player closest to ball"""
        if not players or not ball_detections:
            return None
        
        ball = ball_detections[0]
        ball_center = np.array([
            (ball['bbox'][0] + ball['bbox'][2]) / 2,
            (ball['bbox'][1] + ball['bbox'][3]) / 2
        ])
        
        closest_player = None
        min_distance = float('inf')
        
        for player in players:
            player_center = np.array([
                (player['bbox'][0] + player['bbox'][2]) / 2,
                (player['bbox'][1] + player['bbox'][3]) / 2
            ])
            
            distance = np.linalg.norm(player_center - ball_center)
           
            if distance < min_distance:
               min_distance = distance
               closest_player = player
       
        return closest_player
   
    def run(self):
       """Main application loop"""
       cap = cv2.VideoCapture(0)
       cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
       cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
       cap.set(cv2.CAP_PROP_FPS, 30)
       
       # Start processing thread
       self.processing_active = True
       processing_thread = threading.Thread(target=self.processing_thread)
       processing_thread.start()
       
       # Player assignment mode
       assignment_mode = False
       selected_team = None
       
       print("\nControls:")
       print("- Press 'p' to enter player assignment mode")
       print("- Press 'a' or 'b' to select team during assignment")
       print("- Click on players to assign them")
       print("- Press 's' to show statistics")
       print("- Press 'q' to quit")
       
       cv2.namedWindow('Basketball Tracker')
       cv2.setMouseCallback('Basketball Tracker', self._mouse_callback)
       self.mouse_click = None
       
       try:
           while True:
               ret, frame = cap.read()
               if not ret:
                   break
               
               # Add frame to processing queue
               if self.frame_queue.qsize() < 20:
                   self.frame_queue.put(frame.copy())
               
               # Get processed results
               display_frame = frame
               try:
                   result = self.result_queue.get_nowait()
                   display_frame = self._draw_results(result)
                   
                   # Handle player assignment clicks
                   if assignment_mode and self.mouse_click and selected_team:
                       clicked_player = self._get_clicked_player(
                           result['tracked_players'], 
                           self.mouse_click
                       )
                       if clicked_player:
                           player_name = input(f"Enter player name for {selected_team}: ")
                           self.stats_engine.assign_player(
                               clicked_player['track_id'],
                               selected_team,
                               player_name
                           )
                           print(f"Assigned {player_name} to {selected_team}")
                       self.mouse_click = None
                       
               except queue.Empty:
                   pass
               
               # Draw UI overlay
               self._draw_ui_overlay(display_frame, assignment_mode, selected_team)
               
               # Update FPS
               self.fps_counter += 1
               current_time = time.time()
               if current_time - self.last_fps_time > 1.0:
                   fps = self.fps_counter / (current_time - self.last_fps_time)
                   self.fps_counter = 0
                   self.last_fps_time = current_time
                   cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
               
               cv2.imshow('Basketball Tracker', display_frame)
               
               # Handle keyboard input
               key = cv2.waitKey(1) & 0xFF
               if key == ord('q'):
                   break
               elif key == ord('p'):
                   assignment_mode = not assignment_mode
                   print(f"Assignment mode: {'ON' if assignment_mode else 'OFF'}")
               elif key == ord('a') and assignment_mode:
                   selected_team = 'team_a'
                   print("Selected Team A")
               elif key == ord('b') and assignment_mode:
                   selected_team = 'team_b'
                   print("Selected Team B")
               elif key == ord('s'):
                   print("\n" + self.stats_engine.generate_summary())
               
       finally:
           # Cleanup
           self.processing_active = False
           processing_thread.join()
           cap.release()
           cv2.destroyAllWindows()
           
           # Save final statistics
           self.stats_engine.save_stats()
           print("\nFinal Statistics:")
           print(self.stats_engine.generate_summary())
   
    def _mouse_callback(self, event, x, y, flags, param):
       """Handle mouse clicks"""
       if event == cv2.EVENT_LBUTTONDOWN:
           self.mouse_click = (x, y)
   
    def _get_clicked_player(self, players, click_pos):
       """Find player at click position"""
       for player in players:
           x1, y1, x2, y2 = player['bbox']
           if x1 <= click_pos[0] <= x2 and y1 <= click_pos[1] <= y2:
               return player
       return None
   
    def _draw_results(self, result):
       """Draw detection and tracking results"""
       frame = result['frame'].copy()
       
       # Draw all detections
       for det in result['detections']:
           x1, y1, x2, y2 = det['bbox']
           
           if det['class'] == 'person':
               # Check if assigned
               track_id = det.get('track_id')
               player_info = self.stats_engine.player_assignments.get(track_id)
               
               if player_info:
                   # Assigned player - show name and team color
                   color = (255, 0, 0) if player_info['team'] == 'team_a' else (0, 0, 255)
                   label = player_info['name']
               else:
                   # Unassigned player
                   color = (0, 255, 0)
                   label = f"Player {track_id}"
               
               cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
               cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
               
           elif det['class'] == 'sports ball':
               # Draw ball in yellow
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
               cv2.putText(frame, "Ball", (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
       
       # Draw detected actions
       for action in result['actions']:
           x1, y1, x2, y2 = action['bbox']
           cv2.putText(frame, action['action'].upper(), 
                      (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, (255, 255, 0), 2)
       
       # Shot notification
       if result['shot_detected']:
           cv2.putText(frame, "SHOT DETECTED!", (frame.shape[1]//2 - 100, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
       
       return frame
   
    def _draw_ui_overlay(self, frame, assignment_mode, selected_team):
       """Draw UI information overlay"""
       h, w = frame.shape[:2]
       
       # Status bar background
       cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
       
       # Mode indicator
       if assignment_mode:
           mode_text = f"ASSIGNMENT MODE - Team: {selected_team or 'None selected'}"
           cv2.putText(frame, mode_text, (10, h-35),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
       
       # Quick stats
       team_a_stats = self.stats_engine.get_team_stats('team_a')
       team_b_stats = self.stats_engine.get_team_stats('team_b')
       
       stats_text = f"Team A: {team_a_stats.get('shots_made', 0)}/{team_a_stats.get('shots_attempted', 0)} | "
       stats_text += f"Team B: {team_b_stats.get('shots_made', 0)}/{team_b_stats.get('shots_attempted', 0)}"
       
       cv2.putText(frame, stats_text, (10, h-10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

if __name__ == "__main__":
   tracker = BasketballTracker()
   tracker.setup()
   tracker.run()

   