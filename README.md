# Handwritten Digit Recognition with Conventional Neural Network(CNN)

This project implements a convolutional neural network (CNN) to recognize handwritten digits from the MNIST dataset. The model is trained using Keras with TensorFlow as the backend. The trained model is then deployed using Streamlit, which allows users upload a image of a digit and get a prediction from the model.

Use the app - https://mubarakmayyeri-digit-recognition-cnn-main-syl3gi.streamlit.app/

![alt text](https://github.com/mubarakmayyeri/digit-recognition-CNN/blob/master/assets/images/MNIST-handwritten-digits-dataset-visualized-by-Activeloop.webp "Digit Recognition with CNN") 


## Dataset
The MNIST dataset contains 70,000 grayscale images of handwritten digits, each of size 28x28 pixels. The dataset is split into 60,000 training images and 10,000 test images.

## Model Architecture

The model is a convolutional neural network with the following architecture:

1. Input layer (28x28x1)
2. Convolutional layer with 32 filters and a kernel size of 3x3, followed by a ReLU activation function
3. Convolutional layer with 64 filters and a kernel size of 3x3, followed by a ReLU activation function
4. Max pooling layer with a pool size of 2x2
5. Flatten layer to convert the 2D feature maps into a 1D feature vector
6. Fully connected layer with 128 neurons and a ReLU activation function
7. Output layer with 10 neurons and a softmax activation function


The model is trained using categorical cross-entropy loss and the Adam optimizer.

## Deployment
The trained model is deployed using Streamlit, which is an open-source Python library that makes it easy to create web apps for machine learning and data science. The app allows users to upload an image of a digit and then get a prediction from the model.


## How to use
To make a local copy of the repository and getting it up and running, follow the steps:

### Prerequisites
* Python (I've used Python v3.10.7)
* pip (The Python Package Installer)
* virtualenv package `pip install virtualenv`

### Installation
1. Clone the repo and change the working directory.
   ```
   git clone https://github.com/mubarakmayyeri/digit-recognition-CNN.git
   cd digit-recognition-CNN
   ```
2. Create and activate the virtual environment to avoid any conflicts with the installed packages.
   ```
   virtualenv env
   env\scripts\activate
   ```
3. Install the required packages.
   ```
   pip install -r requirements.txt
   ```
7. Run the streamlit app.
   ```
   streamlit run main.py

## Demo of the model

**Home** 

![alt text](https://github.com/mubarakmayyeri/digit-recognition-CNN/blob/master/assets/images/home.jpg "home_page")  

**Upload image of a digit**  

![alt text](https://github.com/mubarakmayyeri/digit-recognition-CNN/blob/master/assets/images/upload.jpg "upload")  

**Sample Prediction**  


![alt text](https://github.com/mubarakmayyeri/digit-recognition-CNN/blob/master/assets/images/prediction.jpg "Prediction")

## Resources Used

* **Udemy course** - https://www.udemy.com/course/machine-learning-deep-learning-projects-for-beginners-2023/
* **Streamlit Documentation** - https://docs.streamlit.io/library/get-started/create-an-app

