import os
from PIL import Image

# This script contains the code for combining source images and their target masks for training the model

# Define the paths to the two folders containing images
folder_img = 'Add path to source image files'
folder_mask = 'Add path to source image files'

# Define the output folder where the combined images will be saved
folder_out = 'Add path to source image files'

def combine_images(image_folder, mask_folder, output_folder):
    # Ensure the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of image files from each folder
    image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.jpeg', 'png'))]
    mask_files = [os.path.join(mask_folder, filename) for filename in os.listdir(mask_folder) if filename.endswith(('.jpg', '.jpeg', 'png'))]

    # Ensure there are the same number of images in both folders
    if len(image_files) != len(mask_files):
        print("Folders have different numbers of images and cannot be combined.")
    else:
        # Loop through the image files and combine and save them
        for i, (file1, file2) in enumerate(zip(image_files, mask_files)):
            # Open images from both folders
            img1 = Image.open(file1)
            img2 = Image.open(file2)
            # Ensure both images have the same size
            if img1.size != img2.size:
                print(f"Images {file1} and {file2} have different sizes and cannot be combined.")
            else:
                # Combine the two images horizontally (side by side)
                combined_img = Image.new('RGB', (img1.width + img2.width, img1.height))
                combined_img.paste(img1, (0, 0))
                combined_img.paste(img2, (img1.width, 0))

                # Define the filename for the saved image (customize as needed)
                filename = f"combined_image_{i+1}.jpg"

                # Save the combined image to the output folder
                combined_img.save(os.path.join(output_folder, filename))

        print("Combined images saved to the output folder.")

img_combined = combine_images(folder_img, folder_mask, folder_out)