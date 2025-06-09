# action_recognition.py
import cv2
import numpy as np
from collections import deque

class ActionRecognizer:
    def __init__(self):
        self.player_histories = {}
        self.action_buffer_size = 15  # 0.5 seconds at 30fps
        
    def update(self, tracked_players, ball_detection):
        """Analyze player movements for actions"""
        detected_actions = []
        
        for player in tracked_players:
            track_id = player.get('track_id')
            if track_id is None:
                continue
            
            # Initialize history if new player
            if track_id not in self.player_histories:
                self.player_histories[track_id] = {
                    'positions': deque(maxlen=self.action_buffer_size),
                    'has_ball': deque(maxlen=self.action_buffer_size),
                    'last_action': None,
                    'action_cooldown': 0
                }
            
            history = self.player_histories[track_id]
            
            # Update position history
            x1, y1, x2, y2 = player['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            history['positions'].append(center)
            
            # Check ball possession
            has_ball = self._check_ball_possession(player['bbox'], ball_detection)
            history['has_ball'].append(has_ball)
            
            # Decrease cooldown
            if history['action_cooldown'] > 0:
                history['action_cooldown'] -= 1
            
            # Detect actions if enough history
            if len(history['positions']) >= 10 and history['action_cooldown'] == 0:
                action = self._detect_player_action(history)
                if action:
                    detected_actions.append({
                        'track_id': track_id,
                        'action': action,
                        'bbox': player['bbox']
                    })
                    history['last_action'] = action
                    history['action_cooldown'] = 30  # 1 second cooldown
        
        return detected_actions
    
    def _check_ball_possession(self, player_bbox, ball_detection):
        """Check if player has ball possession"""
        if not ball_detection:
            return False
        
        ball = ball_detection[0] if isinstance(ball_detection, list) else ball_detection
        
        # Expand player bbox for possession detection
        p_x1, p_y1, p_x2, p_y2 = player_bbox
        margin = 20
        
        # Get ball center
        b_x1, b_y1, b_x2, b_y2 = ball['bbox']
        ball_center_x = (b_x1 + b_x2) // 2
        ball_center_y = (b_y1 + b_y2) // 2
        
        # Check if ball is within expanded player area
        return (p_x1 - margin <= ball_center_x <= p_x2 + margin and
                p_y1 - margin <= ball_center_y <= p_y2 + margin)
    
    def _detect_player_action(self, history):
        """Detect specific player actions"""
        positions = list(history['positions'])
        has_ball = list(history['has_ball'])
        
        # Calculate movement
        total_movement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_movement += np.sqrt(dx**2 + dy**2)
        
        avg_movement = total_movement / len(positions)
        
        # Detect dribbling (has ball + moving)
        if sum(has_ball[-5:]) >= 3 and avg_movement > 5:
            return 'dribbling'
        
        # Detect traveling (moving without dribbling)
        if sum(has_ball[-10:]) >= 8 and avg_movement > 10:
            # Check for continuous movement without dribble
            return 'travel'
        
        # More actions can be added here
        
        return None
    


    