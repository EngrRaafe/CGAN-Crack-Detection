import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import display
from os import listdir
from numpy import asarray
from keras.utils.image_utils import img_to_array, load_img
import numpy as np
import cv2

# load the dataset
PATH = 'Add path to the training dataset'

# method to load the image files
def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a crack image image
  # - one with the crack's mask image 
  w = tf.shape(image)[1]
  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]
  
  # Convert both images to float32 tensors
  real_image = tf.cast(real_image, tf.float32)
  input_image = tf.cast(input_image, tf.float32)
  
  return real_image, input_image

# Define the buffer size as the number of training samples
BUFFER_SIZE = 9899
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

# methods to perform preprocessing on the loaded images
def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

# Normalizing the images to [-1, 1]
def normalize(real_image, input_image):
  real_image = (real_image / 127.5) - 1
  input_image = (input_image / 127.5) - 1
  
  return real_image,input_image

def load_image_test(image_file):
  real_image,input_image = load(image_file)
  real_image,input_image = resize(real_image,input_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  real_image,input_image = normalize(real_image,input_image)

  return real_image,input_image

# building the testing pipeline
try:
  test_dataset = tf.data.Dataset.list_files('Add path to the testing dataset')
except tf.errors.InvalidArgumentError:
  test_dataset = tf.data.Dataset.list_files('Add path to the testing dataset')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

# load the model
generator_pt = tf.keras.models.load_model('D:/CPEC Project Files/CPEC Source Files/models/Tensorflow_models_9899/model_800000.h5')

def block_ext(img):
    block_size = 32
    block_array = []
   
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            block = img[i:i+block_size, j:j+block_size]
            block_array.append(block)
    return block_array

def add_values_at_indices(original_array, index_array,number):
    for index in index_array:
        original_array[index] = number

    return original_array

def add_values_from_array(new_array, index_array, insert_values):
    for i, index in enumerate(index_array):
        new_array[index] = insert_values[i]

    return new_array

def block_evaluation(model, test_input, tar):

  prediction = model(test_input, training=True) 
  translated_image = (prediction[0].numpy() + 1.0) * 127.5
  converted_image = np.uint8(translated_image)

  bgr = cv2.cvtColor(converted_image, cv2.COLOR_RGB2BGR)
  gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
  _, thresh_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

  translated_mask = (tar[0].numpy() + 1.0) * 127.5
  converted_mask = np.uint8(translated_mask)

  bgr_mask = cv2.cvtColor(converted_mask, cv2.COLOR_RGB2BGR)
  gray_mask = cv2.cvtColor(bgr_mask, cv2.COLOR_BGR2GRAY)
  _, bw_mask = cv2.threshold(gray_mask, 127, 255, cv2.THRESH_BINARY)

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Training Mask', 'Predicted Image']
  plt.figure(figsize=(9, 9))
  
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()    

  # divide the image into 64 blocks of 32 x 32 pixels
  image = block_ext(thresh_image)        
  template = block_ext(bw_mask)
  
  both_black_pixels = []
  template_black_pixels = []
  template_white_pixels = []
  comparison_pixels = []
  iou_array = []
  true_positive = 0
  false_negative = 0
  false_positive = 0
  result = np.zeros((64), dtype=float)

  for i in range(0, len(image)):
    # if both image and template blocks do not have white pixels, 0 correlation value is registered
    if cv2.countNonZero(template[i]) == 0 and cv2.countNonZero(image[i]) == 0:
        both_black_pixels.append(i)
    # if template has black pixels and the image has white pixels, 0 correlation value is registered 
    elif cv2.countNonZero(template[i]) == 0 and cv2.countNonZero(image[i]) > 0:
        template_black_pixels.append(i)
    # if template has white pixels and the image doesn't, correlation value of 0 is registered 
    elif cv2.countNonZero(template[i]) > 0 and cv2.countNonZero(image[i]) == 0:
        template_white_pixels.append(i)
    elif cv2.countNonZero(image[i]) > 0 and cv2.countNonZero(template[i]) > 0:
        comparison_pixels.append(i)
        gen_white_pixel_count = cv2.countNonZero(image[i])
        temp_white_pixel_count = cv2.countNonZero(template[i])
        if gen_white_pixel_count > temp_white_pixel_count:
          true_positive = temp_white_pixel_count
          false_positive = gen_white_pixel_count - temp_white_pixel_count
          false_negative = 0

        elif gen_white_pixel_count < temp_white_pixel_count:
          true_positive = gen_white_pixel_count
          false_negative = temp_white_pixel_count - gen_white_pixel_count
          false_positive = 0
         
        denominator = true_positive + false_negative + false_positive
        iou = abs(true_positive/denominator)
        iou_array.append(iou)

    result_array = add_values_at_indices(result, both_black_pixels, 1.0)
    result_array = add_values_from_array(result_array, comparison_pixels, iou_array)
    crack_sim_ind = np.average(result_array)
    return crack_sim_ind
  
metric_list = []
for inp, tar in test_dataset.take(5):
  metric = block_evaluation(generator_pt, inp, tar)
  metric_list.append(metric)


  
        
