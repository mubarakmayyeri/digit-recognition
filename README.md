# Handwritten Digit Recognition with Conventional Neural Network(CNN)

In this project a CNN Deep Learning model is trained with MNIST dataset for handwritten digit recognition.

![alt text](https://github.com/mubarakmayyeri/digit-recognition-CNN/blob/master/assets/images/MNIST-handwritten-digits-dataset-visualized-by-Activeloop.webp "Digit Recognition with CNN") 

The model was deployed using Streamlit framework.

## Getting Started
To make a local copy of the repository and getting it up and running, follow the steps:

### Prerequisites
* Python (I've used Python v3.10.7)
* pip (The Python Package Installer)
* virtualenv package `pip install virtualenv`

### Installation
1. Clone the repo and change the working directory.
   ```sh
   git clone https://github.com/mubarakmayyeri/digit-recognition-CNN.git
   cd digit-recognition-CNN
   ```
2. Create and activate the virtual environment to avoid any conflicts with the installed packages.
   ```sh
   virtualenv env
   env\scripts\activate
   ```
3. Install the required packages.
   ```sh
   pip install -r requirements.txt
   ```
7. Run the streamlit app.
   ```sh
   streamlit run main.py

## Demo of the model

**Home** 

![alt text](https://github.com/mubarakmayyeri/digit-recognition-CNN/blob/master/assets/images/home.jpg "home_page")  

**Upload image of a digit**  

![alt text](https://github.com/mubarakmayyeri/digit-recognition-CNN/blob/master/assets/images/upload.jpg "upload")  

**Sample Predciction**  


![alt text](https://github.com/mubarakmayyeri/digit-recognition-CNN/blob/master/assets/images/prediction.jpg "Prediction")

## Resources Used

* **Udemy course** - https://www.udemy.com/course/machine-learning-deep-learning-projects-for-beginners-2023/
* **Streamlit Documentaion** - https://docs.streamlit.io/library/get-started/create-an-app

