import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from PIL import Image
from pathlib import Path
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

print("Done with library declaration, Current version of Tensorflow is: ", tf.__version__)


# Function to reduce the size and grayscale it
def image_processing(original_dir_path, grayscale_dir_path):
    file_listing = os.listdir(original_dir_path)
    for file in file_listing:
        im = Image.open(original_dir_path + '\\' + file)
        img = im.resize(image_size)
        gray = img.convert('L')
        gray.save(grayscale_dir_path + '\\' + file, "JPEG")
    pass


# Load and transform data.

# Setting the image size for the processing
image_size = (128, 128)

# collect directory
data_dir = Path(r'Garbage\original_images')

transformer = T.Compose([T.Resize(image_size), T.ToTensor()])
dataset = ImageFolder(data_dir, transform=transformer)

# classes list
class_names = dataset.classes

# Defining the path
PATH_TEST = r"Garbage\original_images"
PATH_TRAIN = r"Garbage\processed_images"

# Image processing for all classes
for idx in class_names:
    class_path = PATH_TEST + "\\" + idx
    process_path = PATH_TRAIN + "\\" + idx
    image_processing(class_path, process_path)
    print("Image processing for " + idx + " is done!")

# Train and test dataset allocation
train_dir = os.path.join(PATH_TRAIN)
test_dir = os.path.join(PATH_TEST)

