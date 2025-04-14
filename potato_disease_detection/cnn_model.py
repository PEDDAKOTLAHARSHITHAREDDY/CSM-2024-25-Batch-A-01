import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Constants
IMAGE_SIZE = (256, 256)  # Updated to match model input shape

BATCH_SIZE = 32
EPOCHS = 50

def create_model(input_shape=(256, 256, 3), num_classes=3):
    """Creates the actual CNN model used in production"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), strides=2, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def get_model_summary(model_path='potato_disease_mobilenet_model.h5'):
    """Load and return model summary"""
    model = tf.keras.models.load_model(model_path)
    model.summary()
    return model

def train_model(data_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    num_classes = len(train_generator.class_indices)
    model = create_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes)
    
    model.fit(train_generator,
              steps_per_epoch=train_generator.samples // BATCH_SIZE,
              validation_data=validation_generator,
              validation_steps=validation_generator.samples // BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1)

    
    model.save('potato_disease_mobilenet_model.h5')

    return model, train_generator.class_indices

if __name__ == '__main__':
    data_dir = 'dataset'  # Path to the image dataset


    train_model(data_dir)
