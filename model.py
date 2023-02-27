# Loading libraries
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import pickle


#l Loading the dataset
def get_data():
    from tensorflow.keras.datasets import mnist
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    return X_train, y_train, X_test, y_test

# preprocessing the data
def preprocess(data, shape):
    rows = shape[0]
    dim_1 = shape[1]
    dim_2 = shape[2]
    dim_3 = shape[3]
    data = data/255
    data = data.reshape(rows, dim_1, dim_2, dim_3)
    
    return data

# Building the model
def build_model(filters_1: int, filters_2: int, kernel_size: tuple, activation: str, input_shape: tuple,
                pool_size: tuple, rate: int, units_1: int, units_2: int, activation_dense: int):
    
    model = Sequential()
    
    # Adding first CNN layer
    model.add(Conv2D(filters=filters_1, kernel_size=kernel_size, activation=activation, input_shape=input_shape))
    
    # Adding second CNN layer
    model.add(Conv2D(filters=filters_2, kernel_size=kernel_size, activation=activation))
    
    # Adding maxpool layer
    model.add(MaxPool2D(pool_size=pool_size))
    
    # Adding dropout layer
    model.add(Dropout(rate))
    
    # Adding flatten layer
    model.add(Flatten())
    
    # Adding fully connected layer
    model.add(Dense(units=units_1, activation=activation))
    
    # Adding output layer
    model.add(Dense(units=units_2, activation=activation_dense))
    
    # Compile the model
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    
    
    
    return model


# Train the model
def train_model(model, X_train, y_train, X_test, y_test):
    print('Model training started.....')
    history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))
    print('Model training finished!!!')
    
    return history

# Model evaluation
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_pred, y_test)
    acc = accuracy_score(y_pred, y_test)
    
    return cm, acc



### RUNNING THE MODEL


if __name__ == '__main__':
    # Collecting train and test sets    
    X_train, y_train, X_test, y_test = get_data()

    # Preprocess train and test dataset
    train_shape = (60000, 28, 28, 1)
    test_shape = (10000, 28, 28, 1)
    X_train = preprocess(X_train, train_shape)
    X_test = preprocess(X_test,test_shape)

    print(X_train.shape)

    # Building the model
    model = build_model(filters_1=32, filters_2=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1),
                        pool_size=(2, 2), rate=(0.4), units_1=128, units_2=10, activation_dense='softmax')


    # Train the model
    train_model(model, X_train, y_train, X_test, y_test)


    # evaluating the model
    '''cm, acc = evaluate(model, X_test, y_test)

    print('Confution Matrix')
    print(cm)
    print(f'Accuracy Score of model is: {acc * 100}%')'''

    # Serializing the model using pickle
    # import pickle
    # pickle_out = open('model.pkl', 'wb')
    # pickle.dump(model, pickle_out)
    # pickle_out.close()
    
    # Saving model
    model.save('trained_model.h5')
