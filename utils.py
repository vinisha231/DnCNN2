import os
import tensorflow as tf
import numpy as np

def load_images_from_folder(folder_path):
    """
    Load images from a folder into TensorFlow tensors.

    Args:
    - folder_path (str): Path to the folder containing images.

    Returns:
    - images (tf.Tensor): TensorFlow tensor containing loaded images.
    """
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png'):
            img = tf.io.read_file(os.path.join(folder_path, filename))
            img = tf.image.decode_png(img, channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            images.append(img)
    images = tf.stack(images)  # Stack the images into a single tensor
    return images

def save_image(image, file_path):
    """
    Save a single image tensor to file.

    Args:
    - image (tf.Tensor): TensorFlow tensor representing the image.
    - file_path (str): File path to save the image.
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.squeeze(image, axis=-1)  # Remove the single-channel dimension if present
    encoded_image = tf.image.encode_png(image)
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with tf.io.gfile.GFile(file_path, 'wb') as f:
        f.write(encoded_image.numpy())

def save_images(images, file_paths):
    """
    Save multiple image tensors to files.

    Args:
    - images (List[tf.Tensor]): List of TensorFlow tensors representing images.
    - file_paths (List[str]): List of file paths to save the images.
    """
    for image, file_path in zip(images, file_paths):
        save_image(image, file_path)

def create_dir_if_not_exist(directory):
    """
    Create a directory if it does not exist.

    Args:
    - directory (str): Directory path to create.
    """
    os.makedirs(directory, exist_ok=True)
