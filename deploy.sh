#!/bin/bash
# deploy.sh - Deployment and setup script

echo "Basketball Tracker Deployment Script"
echo "===================================="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo "Installing dependencies..."
sudo apt install -y python3-pip python3-opencv python3-numpy
sudo apt install -y libatlas-base-dev libhdf5-dev
sudo apt install -y git cmake build-essential

# Install Python packages
echo "Installing Python packages..."
pip3 install ultralytics==8.0.200
pip3 install tflite-runtime
pip3 install numpy opencv-python

# Create project directory
mkdir -p ~/basketball_tracker
cd ~/basketball_tracker

# Download optimized model
echo "Downloading optimized model..."
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Create systemd service for auto-start
echo "Creating systemd service..."
sudo tee /etc/systemd/system/basketball-tracker.service > /dev/null <<EOF
[Unit]
Description=Basketball Stat Tracker
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/basketball_tracker
ExecStart=/usr/bin/python3 /home/pi/basketball_tracker/main_app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Enable service
sudo systemctl daemon-reload
sudo systemctl enable basketball-tracker.service

echo "Deployment complete!"
echo "Start the tracker with: sudo systemctl start basketball-tracker"