'''
# 2 Build the model (CNN)
model = Sequential([
    Input(shape=(D,)),
    Dense(128, activation='relu'),
    Dense(K, activation='softmax'),
])
'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.datasets import fashion_mnist
'''
# 1. Load the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data to [0, 1] and reshape  Nx28x28x1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]  # reshape to (60000, 28, 28, 1)
x_test = x_test[..., tf.newaxis]    # reshape to (10000, 28, 28, 1)

K=len(set(y_train))     # number of classes

# 2. Build the model (CNN for fashion_mnist)
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), strides=2, activation='relu'),
    Conv2D(64, (3, 3), strides=2, activation='relu'),
    Conv2D(128, (3, 3), strides=2, activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(K, activation='softmax'),  # 10 classes for fashion_mnist
])
# Functional API - model = Model(inputs=[i1, i2, i3], outputs=[o1, o2, o3])

# 3 Train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

# 4 Evaluate - plot loss per iteration
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()


### Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
'''
### Plot confusion matrix
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Conf Mat',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print("Normalized conf matrix")
    else:
        print("Conf matrix, without normalization")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

labels = '''T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot'''.split()
'''
### You can also plot some missclassified samples
misclass = np.where(p_test != y_test)[0]
i = np.random.choice(misclass)
plt.imshow(x_test[i].reshape(28,28), cmap='gray')
plt.title("True label: %s Predicted %s" % (labels[y_test[i]], labels[p_test[i]]));
'''

# 5 Make predictions

# 6 data augmentation (virtually adding more data without having it - mirror images...)

# model.summary()


from keras.datasets import cifar10

# 1. Load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()
### Shape x (50000, 32, 32, 3) ; y (50000, )

K= len(set(y_train))

# 2. Build the model (CNN for cifar10)
model = Sequential([
    Input(shape=(32, 32, 3)), # x_train[0].shape
    Conv2D(32, (3, 3), strides=2, activation='relu'),
    Conv2D(64, (3, 3), strides=2, activation='relu'),
    Conv2D(128, (3, 3), strides=2, activation='relu'),
    Flatten(),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.2),
    Dense(K, activation='softmax'),  # 10 classes for cifar10
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

# 4 Evaluate - plot loss per iteration
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()


### Plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')


# model.summary()
plt.show()