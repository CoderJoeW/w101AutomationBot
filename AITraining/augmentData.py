import os
import shutil
from PIL import Image
import imgaug.augmenters as iaa
import numpy as np

# Path to the source folder where images are stored
source_folder = 'training'

# Path to the destination folder where images will be organized and augmented
destination_folder = 'set1'

# Define a more complex sequence of augmentations
seq = iaa.Sequential([
    iaa.SomeOf((2, 5), [  # Apply 2 to 5 of the following augmentations randomly
        iaa.Fliplr(0.5),  # 50% chance to horizontally flip images
        iaa.Flipud(0.2),  # 20% chance to vertically flip images
        iaa.Affine(rotate=(-25, 25)),  # Rotate images between -25 and 25 degrees
        iaa.Multiply((0.8, 1.2)),  # Change brightness (80-120%)
        iaa.GaussianBlur(sigma=(0, 1.0)),  # Apply Gaussian blur with a sigma of 0 to 1.0
        iaa.LinearContrast((0.75, 1.5)),  # Improve or worsen the contrast
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Add gaussian noise
        iaa.Crop(percent=(0, 0.1)),  # Random crops
    ]),
    iaa.Resize({"height": "keep-aspect-ratio", "width": 0.9})  # Resize width to 90% keeping aspect ratio
], random_order=True)  # Apply the augmentations in random order

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# List all files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file is an image
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Extract the direction part from the filename
        direction = filename.split('_')[-1].split('.')[0]

        # Path to the subfolder for this direction
        direction_folder = os.path.join(destination_folder, direction)

        # Ensure the subfolder for this direction exists
        os.makedirs(direction_folder, exist_ok=True)

        # Construct the full paths for source and destination
        source_path = os.path.join(source_folder, filename)

        # Load the image
        image = Image.open(source_path)
        image_np = np.array(image)  # Convert the image to a numpy array

        # Apply augmentations
        images_aug = [seq(image=image_np) for _ in range(30)]  # Generate 10 augmented versions

        # Save the augmented images
        for idx, img_aug in enumerate(images_aug):
            aug_image = Image.fromarray(img_aug)
            aug_filename = f"{os.path.splitext(filename)[0]}_aug{idx}{os.path.splitext(filename)[1]}"
            destination_path = os.path.join(direction_folder, aug_filename)
            aug_image.save(destination_path)

print("Images have been organized and augmented in subfolders by direction.")
