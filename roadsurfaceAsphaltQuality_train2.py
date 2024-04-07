import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np

# Adding Seed so that random initialization is consistent
seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)

batch_size = 32

# Prepare input data
classes = os.listdir('training_data_asphalt_quality')
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path = 'training_data_asphalt_quality'

# We shall load all the training and validation images and labels into memory using TensorFlow data loaders
#data = tf.keras.utils.get_file(
 #   'dataset.zip',
  #  'https://example.com/dataset.zip',
   # extract=True,
#)
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=validation_size,
    subset="training",
    seed=seed,
    image_size=(img_size, img_size),
    batch_size=batch_size,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    validation_split=validation_size,
    subset="validation",
    seed=seed,
    image_size=(img_size, img_size),
    batch_size=batch_size,
)

# Normalize pixel values to be between 0 and 1
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, num_channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_ds, validation_data=val_ds, epochs=5)

# Save the model in SavedModel format
model.save('./saved_model/roadsurfaceAsphaltQuality-model')
