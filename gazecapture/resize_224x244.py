import os
from PIL import Image
import shutil


def resize_images(source_dir, target_dir, folder_name):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            # Create the corresponding directory in the target directory
            target_subdir = os.path.join(
                target_dir, os.path.relpath(os.path.join(root, dir_name), source_dir)
            )
            os.makedirs(target_subdir, exist_ok=True)
            """By setting exist_ok as True
            error caused due already
            existing directory can be suppressed
            but other OSError may be raised
            due to other error like
            invalid path name """

        for file_name in files:
            # Only process .jpg files
            if file_name.endswith(".jpg"):
                source_file_path = os.path.join(root, file_name)
                target_file_path = os.path.join(
                    target_dir, os.path.relpath(source_file_path, source_dir)
                )
                if root.endswith(folder_name):
                    # Open and resize the image
                    img = Image.open(source_file_path)
                    resized_img = img.resize((224, 224), Image.ANTIALIAS)

                    # Save the resized image to the target directory
                    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                    resized_img.save(target_file_path)
                else:
                    # Open and resize the image
                    img = Image.open(source_file_path)

                    # Save the resized image to the target directory
                    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                    img.save(target_file_path)


# Define the source and target directories
source_dir = r"C:\Users\anja.kovacevic\datasets_and_models\B"
target_dir = r"C:\Users\anja.kovacevic\datasets_and_models\C"

folder_name = "appleFace"

# Call the function
resize_images(source_dir, target_dir, folder_name)


"""
# Druga opcija posto pise da ANTIALIAS nestaje u buducem izdanju 

import cv2

# Load the image
img = cv2.imread('input.jpg')

# Resize the image
resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

# Save the image
cv2.imwrite('output.jpg', resized_img)
"""
