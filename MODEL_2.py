import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils

#from keras.datasets import mnist
#(X_train, Y_train) , (X_test , Y_test) = mnist.load_data()


# display(X_train.info())
# display(X_test.info())
# display(X_train.head(n = 2))
# display(X_test.head(n = 2))

X_train = pd.read_csv('sign_mnist_train.csv')
X_test = pd.read_csv('sign_mnist_test.csv')

Y_train = X_train['label']
Y_test = X_test['label']

del X_train['label']
del X_test['label']


X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))


# Normalize the data
X_train = X_train / 255
X_test = X_test / 255


num_classes = 26
y_train = np.array(Y_train).reshape(-1)
y_test = np.array(Y_test).reshape(-1)
y_train = np.eye(num_classes)[Y_train]
y_test = np.eye(num_classes)[Y_test]



classifier = Sequential()
classifier.add(Convolution2D(filters=8, kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(28,28,1),activation='relu', data_format='channels_last'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(filters=16, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(MaxPooling2D(pool_size=(4,4)))
classifier.add(Dense(128, activation='relu'))
classifier.add(Flatten())
classifier.add(Dense(26, activation='softmax'))

classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, epochs=50, batch_size=200)

accuracy = classifier.evaluate(x=X_test,y=y_test,batch_size=32)
print("Accuracy: ",accuracy[1])

handges = classifier.to_json()
with open("hanges.json", "w") as json_file:
    json_file.write(handges)
classifier.save_weights("hange.h5")