# test_system.py
import unittest
import cv2
import numpy as np
from main_app import BasketballTracker

class TestBasketballTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = BasketballTracker()
        
    def test_detection(self):
        """Test object detection functionality"""
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run detection
        detections = self.tracker.detector.detect_frame(test_frame)
        
        # Should return list (even if empty)
        self.assertIsInstance(detections, list)
    
    def test_tracking_consistency(self):
        """Test tracker maintains IDs across frames"""
        # Simulate detections
        frame1_detections = [
            {'bbox': [100, 100, 150, 200], 'class': 'person', 'confidence': 0.9}
        ]
        frame2_detections = [
            {'bbox': [105, 102, 155, 202], 'class': 'person', 'confidence': 0.9}
        ]
        
        # Update tracker
        tracked1 = self.tracker.tracker.update(frame1_detections)
        tracked2 = self.tracker.tracker.update(frame2_detections)
        
        # Should maintain same track_id
        self.assertEqual(tracked1[0]['track_id'], tracked2[0]['track_id'])
    
    def test_stats_recording(self):
        """Test statistics recording"""
        # Assign test player
        self.tracker.stats_engine.assign_player(1, 'team_a', 'Test Player')
        
        # Record shot
        self.tracker.stats_engine.record_shot(1, True, 100, (5.0, 10.0))
        
        # Check stats
        stats = self.tracker.stats_engine.get_player_stats('team_a', 'Test Player')
        self.assertEqual(stats['shots_attempted'], 1)
        self.assertEqual(stats['shots_made'], 1)

def run_integration_test():
    """Run full integration test with camera"""
    print("Running integration test...")
    
    tracker = BasketballTracker()
    tracker.setup()
    
    # Run for 30 seconds
    import threading
    timer = threading.Timer(30.0, lambda: print("\nTest complete!"))
    timer.start()
    
    try:
        tracker.run()
    except KeyboardInterrupt:
        timer.cancel()
    
    print("\nTest Results:")
    print(tracker.stats_engine.generate_summary())

if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False)
    
    # Run integration test
    run_integration_test()