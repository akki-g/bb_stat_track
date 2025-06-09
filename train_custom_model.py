# train_custom_model.py
"""
Optional: Train custom YOLOv8 model on basketball-specific data
Run this on a more powerful machine, then transfer to Pi
"""

from ultralytics import YOLO
import yaml

def create_basketball_dataset_yaml():
    """Create dataset configuration for basketball training"""
    dataset_config = {
        'path': './basketball_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        
        'names': {
            0: 'player',
            1: 'basketball',
            2: 'hoop',
            3: 'referee'
        }
    }
    
    with open('basketball_dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    return 'basketball_dataset.yaml'

def train_basketball_model():
    """Train custom basketball detection model"""
    # Load a pretrained model
    model = YOLO('yolov8n.pt')
    
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
        device='0'  # Use GPU if available
    )
    
    # Export for Raspberry Pi
    model.export(format='tflite', int8=True, imgsz=320)
    
    print("Training complete! Model saved as basketball_model.tflite")

if __name__ == "__main__":
    # Note: You need to prepare your dataset first
    # Structure: basketball_dataset/images/train/, val/, test/
    # With corresponding labels in basketball_dataset/labels/
    train_basketball_model()