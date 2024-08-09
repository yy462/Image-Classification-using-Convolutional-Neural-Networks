import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

input_folder = 'data/cats_vs_dogs/train/'
output_folder = 'data/cats_vs_dogs/split/'

# Create train/val split
train_dir = os.path.join(output_folder, 'train')
val_dir = os.path.join(output_folder, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get list of images
images = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

# Copy images to respective folders
for image in tqdm(train_images, desc="Copying train images"):
    label = 'dog' if 'dog' in image else 'cat'
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    shutil.copy(os.path.join(input_folder, image), os.path.join(train_dir, label, image))

for image in tqdm(val_images, desc="Copying val images"):
    label = 'dog' if 'dog' in image else 'cat'
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)
    shutil.copy(os.path.join(input_folder, image), os.path.join(val_dir, label, image))