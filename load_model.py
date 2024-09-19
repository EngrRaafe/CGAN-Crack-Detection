import numpy as np
from PIL import Image as im
import cv2
import tensorflow as tf
import keras
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#load the model
generator_pt = tf.keras.models.load_model('Add path to the model file')

def load_model(file_path):      
    test_image = im.open(file_path)
    test_array = np.array(test_image)
    input_height = test_array.shape[0]
    input_width = test_array.shape[1]

    # Resize the image to the required input size of your model
    target_size = (256, 256)
    resized_image = tf.image.resize(test_array, target_size)

    # Normalize pixel values to the range [-1, 1] if that's what your model expects
    norm_image = (resized_image / 127.5) - 1.0
    norm_image = tf.expand_dims(norm_image, axis=0)

    # Pass the image to the model
    generated_mask = generator_pt(norm_image, training=True)
    translated_mask = (generated_mask[0].numpy() + 1.0) * 127.5
    output_array = np.uint8(translated_mask)
        
    # apply thresholding to convert generated mask to b/w
    bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # reshaping the image for plotting
    reshape_thresh = np.reshape(threshold_image, (256,256))
    final_array = im.fromarray(reshape_thresh)
        
    # revert the size of the generated mask to original dimensions
    final_image = final_array.resize((input_width, input_height), resample=im.LANCZOS)
    filename = os.path.basename(file_path)
    final_image = final_image.save(filename)
    print(f"Generated Mask saved as {filename}.jpg")
    return final_image

image_path = "Add path to the image for mask generation"

prediction = load_model(image_path)
