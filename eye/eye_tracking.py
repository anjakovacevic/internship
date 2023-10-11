# A modern eye tracking technique utilizes a Convolutional Neural Network (CNN)
# for gaze estimation without requiring calibration, making it suitable for 
# real-world environments. This technique comprises two components: 
# a face component for extracting gaze features from the eyes, and a 
# 39-point facial landmark component for encoding eye shape and location 
# into the network, enhancing the model's learning of head and eye movements. 
# It demonstrated superior performance compared to another model in experiments, 
# especially when fine-tuned with the VGG16 pre-trained model.
# Here's what we did:
# We created a face_component that represents the CNN for extracting gaze features
# from the eyes. We created a facial_landmark_component to handle the 
# 39-point landmarks. In the assemble_model function, we combined both components 
# and concatenated their outputs. We loaded the VGG16 model for fine-tuning. 
# In practice, you'd also want to use its features as inputs to our network.
# To utilize this code, you'd need to further:
# Load your dataset
# Potentially freeze some layers of VGG16 before fine-tuning
# Fine-tune the model on your eye-tracking data
# Evaluate the model's performance
# Note: This is a basic implementation. Depending on your dataset and requirements, you might need to adjust layers, parameters, and training settings for optimal performance.

# https://www.frontiersin.org/articles/10.3389/frai.2021.796825/full
# During fine-tuning, we removed the last fully connected layer of the VGG16 
# network (which has 1,000 classes) and added some convolutional and 
# fully connected layers to the VGG16 network. We also freeze the pre-trained 
# weights of the VGG16 model. Finally, we trained the added layers using the 
# training dataset. During training, the x and y ground truth labels were provided
# to the network.

import tensorflow as tf
from keras import layers, Model
from keras.applications import VGG16

# Face Component: Extracting gaze features from the eyes
def face_component(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(128, activation='relu')(x)
    return Model(inputs, outputs)

# 39-point Facial Landmark Component: Encoding eye shape and location
def facial_landmark_component(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(512, activation='relu')(x)
    return Model(inputs, outputs)

# Assembling the model
def assemble_model(face_input_shape, landmark_input_shape):
    face_inputs = layers.Input(shape=face_input_shape)
    landmark_inputs = layers.Input(shape=landmark_input_shape)
    
    face_model = face_component(face_input_shape)
    landmark_model = facial_landmark_component(landmark_input_shape)
    
    face_outputs = face_model(face_inputs)
    landmark_outputs = landmark_model(landmark_inputs)
    
    concatenated = layers.concatenate([face_outputs, landmark_outputs])
    
    x = layers.Dense(1024, activation='relu')(concatenated)
    final_outputs = layers.Dense(2)(x)  # Assuming the output is 2D gaze coordinates
    
    return Model([face_inputs, landmark_inputs], final_outputs)

# Load VGG16 model and weights for fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# For demonstration purposes, set trainable to True. 
# In practice, you'd likely freeze many of these layers before fine-tuning.
for layer in base_model.layers:
    layer.trainable = True

face_input_shape = (224, 224, 3)  # Example shape for the eye region image
landmark_input_shape = (39, 2)    # 39 2D landmarks

model = assemble_model(face_input_shape, landmark_input_shape)

# You'll need to compile and train this model on your specific data
model.compile(optimizer='adam', loss='mse')  # Mean squared error if it's a regression task

print(model.summary())
