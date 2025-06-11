import os

# Configuration: set your dataset base directory here
BASE_DIR = 'data/test'  # adjust if your dataset path is different

LABELS_DIR = os.path.join(BASE_DIR, 'labels')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')

def clean_empty_labels(base_dir, labels_dir, images_dir):
    removed = []
    # Iterate over each split directory (e.g., train, val, test)
    for split in os.listdir(labels_dir):
        labels_split = os.path.join(labels_dir, split)
        images_split = os.path.join(images_dir, split)
        if not os.path.isdir(labels_split):
            continue
        # Ensure corresponding images directory exists
        if not os.path.isdir(images_split):
            print(f"Warning: No images directory for split '{split}'")
            continue

        for fname in os.listdir(labels_split):
            if not fname.endswith('.txt'):
                continue
            label_path = os.path.join(labels_split, fname)
            # Check if label file is empty or contains only whitespace
            with open(label_path, 'r') as f:
                content = f.read().strip()
            if not content:
                # Remove the empty label file
                os.remove(label_path)
                # Remove corresponding image files (.jpg and .png)
                base_name = os.path.splitext(fname)[0]
                for ext in ['.jpg', '.PNG', '.png']:
                    img_path = os.path.join(images_split, base_name + ext)
                    if os.path.isfile(img_path):
                        os.remove(img_path)
                        removed.append(os.path.join(split, base_name + ext))
                removed.append(os.path.join(split, fname))

    # Report
    if removed:
        print("Removed the following empty labels and images:")
        for path in removed:
            print("  -", path)
    else:
        print("No empty label files found.")

if __name__ == "__main__":
    clean_empty_labels(BASE_DIR, LABELS_DIR, IMAGES_DIR)
