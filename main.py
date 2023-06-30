import os
from pathlib import Path
import tensorflow as tf
import torchvision.transforms as T
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torchvision.datasets import ImageFolder

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
IMG_HEIGHT = 200
IMG_WIDTH = 200
image_size = (IMG_HEIGHT, IMG_WIDTH)

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
# training images are nothing but the processed images
train_dir = os.path.join(PATH_TRAIN)
test_dir = os.path.join(PATH_TEST)

# Image data generator for image transformation
image_gen = ImageDataGenerator(rescale=1. / 255)

# Data generation for training images
train_data_gen = image_gen.flow_from_directory(
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

# Data generation for testing images
test_data_gen = image_gen.flow_from_directory(
    directory=test_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

# Obtaining the class indices of the generated data
train_class_indices = train_data_gen.class_indices
print("Class indices of the generated data:", train_class_indices)

# The number of samples in the generated data in integer format
num_samples = train_data_gen.samples
print("The number of samples in the generated data:", num_samples)

# Generated image shape in tuple format
gen_shape = train_data_gen.image_shape
print("Generated image shape:", gen_shape)

# Defining & Building the CNN model
# Parameter declaration and adding the layers
model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),

    Flatten(),

    Dense(6, activation='softmax')
])

# Defining the model parameters and optimizer
batch_size = 45
epochs = 75
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Print the number of epochs
print("Number of epochs used for building the model: ", epochs)

# Fitting the model
model_fit = model.fit(
    train_data_gen,
    validation_data=train_data_gen,
    steps_per_epoch=num_samples // batch_size,
    epochs=epochs,
    validation_steps=(num_samples - 500) // batch_size,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=7)]
)

# Testing the model against the test data
test_loss, test_acc = model.evaluate(test_data_gen)
print('Test accuracy: {} Test Loss: {} '.format(test_acc * 100, test_loss))

# Getting the accuracy of the model
train_acc = model_fit.history['accuracy']  # store training accuracy in history
final_model_accuracy = train_acc[-1]
print("The accuracy of the fitted model: ", final_model_accuracy)

# Saving the model
model.save('model_200.h5')
