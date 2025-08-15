import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import tensorflow as tf
from tensorboard import program
import datetime


# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.3),  # Add dropout
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.3),  # Add dropout
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
# #callback
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3,restore_best_weights=True)


# #train the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(
#     x_train, y_train,
#     epochs=8,
#     batch_size=32,
#     validation_data=(x_test, y_test),
#     verbose=1,
#     callbacks=[callback]
# )

# model.save('handwritten.keras')
model = tf.keras.models.load_model('handwritten.keras')

# TEST MODEL ON MNIST DATA
print("=== TESTING MODEL ON MNIST DATA ===")
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Test first 10 MNIST images
print("Testing MNIST images:")
for i in range(1000):
    prediction = model.predict(np.array([x_test[i]]), verbose=0)
    predicted = np.argmax(prediction)
    actual = y_test[i]
    confidence = np.max(prediction) * 100
    with open("log.txt", "a") as l:
        l.write(f"MNIST {i}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.1f}%\n")
    print(f"MNIST {i}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.1f}%")
    
    # # Show the MNIST image and prediction
    # plt.figure(figsize=(6, 3))
    # plt.subplot(1, 2, 1)
    # plt.imshow(x_test[i], cmap='gray')
    # plt.title(f'MNIST Image {i}\nActual: {actual}, Predicted: {predicted}')
    # plt.axis('off')
    
    # plt.subplot(1, 2, 2)
    # plt.bar(range(10), prediction[0])
    # plt.title(f'Prediction Probabilities')
    # plt.xlabel('Digit')
    # plt.ylabel('Probability')
    # plt.xticks(range(10))
    
    # plt.tight_layout()
    # # plt.show()


# image_number = 0
# while os.path.isfile(f"digits/digit{image_number}.png"):
#     try:
#         # Your current preprocessing
#         img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
#         img = cv2.resize(img, (28,28))
#         img = np.invert(np.array([img]))
        
#         # Get prediction
#         prediction = model.predict(img, verbose=0)
#         predicted = np.argmax(prediction)
#         confidence = np.max(prediction) * 100
        
#         print(f"Digit {image_number}: Predicted={predicted}, Confidence={confidence:.1f}%")
        
#         # Show your image and prediction
#         plt.figure(figsize=(6, 3))
#         plt.subplot(1, 2, 1)
#         plt.imshow(img[0], cmap='gray')
#         plt.title(f'Digit {image_number}\nPredicted: {predicted}')
#         plt.axis('off')
        
#         plt.subplot(1, 2, 2)
#         plt.bar(range(10), prediction[0])
#         plt.title(f'Prediction Probabilities')
#         plt.xlabel('Digit')
#         plt.ylabel('Probability')
#         plt.xticks(range(10))
        
#         plt.tight_layout()
#         plt.show()
        
#     except Exception as e:
#         print(f"Error processing digit{image_number}.png: {e}")
#     finally:
#         image_number += 1

        
        








