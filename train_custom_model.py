# train_custom_model.py
"""
Optional: Train custom YOLOv8 model on basketball-specific data
Run this on a more powerful machine, then transfer to Pi
"""

from ultralytics import YOLO
import yaml
import os


BASE_DIR = 'data/test'

def create_basketball_dataset_yaml():
    dataset_config = {
        'path': BASE_DIR,
        'train': 'images/train',
        'val':   'images/val',
        'test':  'images/test',
        'nc':    4,   # ← add this here if you’re generating YAML in code
        'names': {
            0: 'basketball',
            1: 'player-team1',
            2: 'player-team2',
            3: 'referee'
        }
    }
    yaml_path = os.path.join(BASE_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f)
    return yaml_path

def train_basketball_model():
    """Train custom basketball detection model"""
    # Load a pretrained model
    model = YOLO('yolov8s.pt')
    
    # Create dataset config
    dataset_yaml = create_basketball_dataset_yaml()
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        name='basketball_model',
        patience=50,
        save=True,
        device='mps'  # Use GPU if available
    )
    
    # Export for Raspberry Pi
    model.export(format='tflite', int8=True, imgsz=320)
    
    print("Training complete! Model saved as basketball_model.tflite")

if __name__ == "__main__":
    # Note: You need to prepare your dataset first
    # Structure: basketball_dataset/images/train/, val/, test/
    # With corresponding labels in basketball_dataset/labels/
    train_basketball_model()

    