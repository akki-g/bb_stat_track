import os
import zipfile
import shutil
import csv
import cv2
import numpy as np
from xml.etree import ElementTree as ET
from sklearn.cluster import KMeans
import yaml

# Paths to dataset archives or folders
ZIPS = {
    'player_detect': 'data/archive (8).zip',
    'ball_tracking': 'data/archive (7).zip',
    'half_court': 'data/Half Court Basketball.v3-resized640_aug15x-firstbatch-labelassist-microsoftcoco.yolov8.zip'
}

# Unified dataset paths
BASE_DIR = 'data/unified_dataset'
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
LABELS_DIR = os.path.join(BASE_DIR, 'labels')
SPLITS = ['train', 'val', 'test']


# Class mapping: (source_dataset, original_class_id) -> unified_class_id
CLASS_MAP = {
    # player_detect classes: 0=BasketBall, 1=Player
    ('player_detect', 0): 0,   # ball
    ('player_detect', 1): -2,  # generic player; will assign by color
    # ball_tracking: only ball class
    ('ball_tracking', 0): 0,
    # half_court original IDs per Roboflow export
    ('half_court', 0): 0,  # ball
    ('half_court', 4): 1,  # player-team1
    ('half_court', 5): 2,  # player-team2
    ('half_court', 6): 3   # referee
}

# Step 1: Unpack all zip archives
def unpack_zips():
    for key, src in ZIPS.items():
        dest = os.path.join(BASE_DIR, key)
        os.makedirs(dest, exist_ok=True)
        if not os.path.exists(src) and src.endswith('.zip'):
            alt = src[:-4]  # try folder with same name
            if os.path.exists(alt):
                src = alt
        if os.path.isdir(src):
            shutil.copytree(src, dest, dirs_exist_ok=True)
        elif os.path.isfile(src):
            with zipfile.ZipFile(src, 'r') as z:
                z.extractall(dest)
        else:
            raise FileNotFoundError(f"Dataset source not found: {src}")
    print("Unpacked all archives.")

# Step 2: Convert ball_tracking CSV & XML to YOLO label txt
def convert_ball_tracking():
    bt_dir = os.path.join(BASE_DIR, 'ball_tracking')
    img_dir = os.path.join(bt_dir, 'boxes')
    # place all frames into a train split
    lbl_dir = os.path.join(bt_dir, 'labels', 'train')
    img_out_dir = os.path.join(bt_dir, 'images', 'train')
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)

    xml_path = os.path.join(bt_dir, 'annotations.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find('.//original_size/width').text)
    height = int(root.find('.//original_size/height').text)

    for box in root.iter('box'):
        if box.get('outside') == '1':
            continue
        frame = int(box.get('frame'))
        xtl = float(box.get('xtl'))
        ytl = float(box.get('ytl'))
        xbr = float(box.get('xbr'))
        ybr = float(box.get('ybr'))

        x_center = ((xtl + xbr) / 2) / width
        y_center = ((ytl + ybr) / 2) / height
        bw = (xbr - xtl) / width
        bh = (ybr - ytl) / height

        img_name = f"frame_{frame:06}.PNG"
        src_img = os.path.join(img_dir, img_name)
        dst_img = os.path.join(img_out_dir, img_name)
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.copy(src_img, dst_img)

        lbl_path = os.path.join(lbl_dir, f"frame_{frame:06}.txt")
        with open(lbl_path, 'w') as lf:
            lf.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
    print("Converted ball_tracking to YOLO format.")

# Step 3: Remap & prune labels, stage for merging
def remap_labels():
    for src in ['player_detect', 'ball_tracking', 'half_court']:
        src_dir = os.path.join(BASE_DIR, src)
        # Identify splits
        for split in SPLITS:
            # Try both common structures
            img_src = os.path.join(src_dir, 'images', split)
            lbl_src = os.path.join(src_dir, 'labels', split)
            if not os.path.isdir(img_src):
                # Maybe flat in src?
                img_src = os.path.join(src_dir, split, 'images')
                lbl_src = os.path.join(src_dir, split, 'labels')
            if not os.path.isdir(img_src):
                continue
            for fname in os.listdir(lbl_src):
                label_path = os.path.join(lbl_src, fname)
                img_fname = fname.replace('.txt', '.jpg')
                img_path = os.path.join(img_src, img_fname)
                if not os.path.exists(img_path):
                    img_fname = fname.replace('.txt', '.PNG')
                    img_path = os.path.join(img_src, img_fname)
                    if not os.path.exists(img_path):
                        continue
                # Read and rewrite label
                unified_lbls = []
                with open(label_path) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        orig_class = int(parts[0])
                        key = (src, orig_class)
                        if key not in CLASS_MAP:
                            continue
                        unified_id = CLASS_MAP[key]
                        if unified_id < 0:
                            unified_id = -2  # generic player
                        unified_lbls.append([unified_id] + parts[1:])
                # Write to unified folder
                out_img_dir = os.path.join(IMAGES_DIR, split)
                out_lbl_dir = os.path.join(LABELS_DIR, split)
                os.makedirs(out_img_dir, exist_ok=True)
                os.makedirs(out_lbl_dir, exist_ok=True)
                # Copy image
                dest_img = os.path.join(out_img_dir, f"{src}_{img_fname}")
                shutil.copy(img_path, dest_img)
                # Write label
                dest_lbl = os.path.join(out_lbl_dir, f"{src}_{fname}")
                with open(dest_lbl, 'w') as outf:
                    for lbl in unified_lbls:
                        outf.write(" ".join(map(str, lbl)) + "\n")
    print("Remapped & pruned labels into unified folders.")

# Step 4: Train color centroids for team jersey assignment
def train_color_centroids():
    half_lbl_dir = os.path.join(BASE_DIR, 'half_court', 'train', 'labels')
    half_img_dir = os.path.join(BASE_DIR, 'half_court', 'train', 'images')
    samples = {4: [], 5: []}
    for fname in os.listdir(half_lbl_dir):
        parts = os.path.splitext(fname)[0]
        with open(os.path.join(half_lbl_dir, fname)) as lf:
            for line in lf:
                cls, xc, yc, w, h = map(float, line.split())
                if int(cls) in [4, 5]:  # original half_court team classes
                    img = cv2.imread(os.path.join(half_img_dir, parts + '.jpg'))
                    h_img, w_img = img.shape[:2]
                    x1 = int((xc - w/2) * w_img)
                    y1 = int((yc - h/2) * h_img)
                    x2 = int((xc + w/2) * w_img)
                    y2 = int((yc + h/2) * h_img)
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    samples[int(cls)].append(crop.reshape(-1, 3))
    # Combine and cluster per team
    centroids = {}
    for team_cls, pix in samples.items():
        pix = np.vstack([p[np.random.choice(len(p), size=500, replace=False)] for p in pix if len(p)>=500])
        kmeans = KMeans(n_clusters=1).fit(pix)
        centroids[team_cls] = kmeans.cluster_centers_[0]
    np.save(os.path.join(BASE_DIR, 'team_centroids.npy'), centroids)
    print("Trained and saved team color centroids.")

# Step 5: Assign generic players to teams by color
def assign_teams():
    centroids = np.load(os.path.join(BASE_DIR, 'team_centroids.npy'), allow_pickle=True).item()
    for split in SPLITS:
        lbl_dir = os.path.join(LABELS_DIR, split)
        img_dir = os.path.join(IMAGES_DIR, split)
        if not os.path.isdir(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            parts = os.path.splitext(fname)[0]
            label_path = os.path.join(lbl_dir, fname)
            with open(label_path) as lf:
                lines = [l.strip().split() for l in lf]
            new_lines = []
            img = None
            for parts_line in lines:
                cls = int(parts_line[0])
                if cls != -2:
                    new_lines.append(parts_line)
                else:
                    # load image crop
                    if img is None:
                        img_path = os.path.join(img_dir, fname.replace('.txt', '.jpg'))
                        if not os.path.exists(img_path):
                            img_path = os.path.join(img_dir, fname.replace('.txt', '.PNG'))
                        img = cv2.imread(img_path)
                    xc, yc, w, h = map(float, parts_line[1:])
                    h_img, w_img = img.shape[:2]
                    x1 = int((xc - w/2) * w_img); y1 = int((yc - h/2) * h_img)
                    x2 = int((xc + w/2) * w_img); y2 = int((yc + h/2) * h_img)
                    crop = img[y1:y2, x1:x2]
                    avg_color = crop.reshape(-1, 3).mean(axis=0)
                    # compare to centroids
                    d1 = np.linalg.norm(avg_color - centroids[4])
                    d2 = np.linalg.norm(avg_color - centroids[5])
                    new_cls = 1 if d1 < d2 else 2
                    new_lines.append([new_cls] + parts_line[1:])
            # overwrite label
            with open(label_path, 'w') as lf:
                for ln in new_lines:
                    lf.write(" ".join(map(str, ln)) + "\n")
    print("Assigned generic players to team1 or team2 based on color centroids.")

# Step 6: Write data.yaml for YOLOv8
def write_data_yaml():
    data = {
        'path': BASE_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 4,
        'names': ['ball', 'player-team1', 'player-team2', 'referee']
    }
    with open(os.path.join(BASE_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(data, f)
    print("Wrote data.yaml for training.")

# Main ETL orchestration
if __name__ == "__main__":
    unpack_zips()
    convert_ball_tracking()
    remap_labels()
    train_color_centroids()
    assign_teams()
    write_data_yaml()
    print("ETL pipeline complete. Unified dataset ready at:", BASE_DIR)
