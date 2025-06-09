# player_tracking.py
import numpy as np
from collections import defaultdict

class SimpleTracker:
    """Simplified ByteTrack-inspired tracker"""
    def __init__(self, max_lost=30):
        self.tracks = {}
        self.track_id_counter = 0
        self.max_lost = max_lost
        
    def update(self, detections):
        """Update tracks with new detections"""
        # Filter for person detections only
        person_detections = [d for d in detections if d['class'] == 'person']
        
        if not self.tracks:
            # Initialize tracks for first frame
            for det in person_detections:
                self.tracks[self.track_id_counter] = {
                    'bbox': det['bbox'],
                    'lost': 0,
                    'history': [det['bbox']]
                }
                det['track_id'] = self.track_id_counter
                self.track_id_counter += 1
        else:
            # Match detections to existing tracks using IoU
            matched = set()
            for det in person_detections:
                best_iou = 0
                best_track_id = None
                
                for track_id, track in self.tracks.items():
                    if track_id in matched:
                        continue
                    
                    iou = self._calculate_iou(det['bbox'], track['bbox'])
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_track_id = track_id
                
                if best_track_id is not None:
                    # Update existing track
                    self.tracks[best_track_id]['bbox'] = det['bbox']
                    self.tracks[best_track_id]['lost'] = 0
                    self.tracks[best_track_id]['history'].append(det['bbox'])
                    det['track_id'] = best_track_id
                    matched.add(best_track_id)
                else:
                    # Create new track
                    self.tracks[self.track_id_counter] = {
                        'bbox': det['bbox'],
                        'lost': 0,
                        'history': [det['bbox']]
                    }
                    det['track_id'] = self.track_id_counter
                    self.track_id_counter += 1
            
            # Update lost count for unmatched tracks
            tracks_to_remove = []
            for track_id, track in self.tracks.items():
                if track_id not in matched:
                    track['lost'] += 1
                    if track['lost'] > self.max_lost:
                        tracks_to_remove.append(track_id)
            
            # Remove lost tracks
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
        
        return person_detections
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    