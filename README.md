# Pneumonia Classifier Project

This project focuses on developing a robust machine learning and deep learning-based classifier to distinguish between normal and pneumonia-affected X-ray images. The goal is to aid in the timely and accurate diagnosis of pneumonia, which is critical for initiating treatment and reducing complications.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Models](#deep-learning-models)
- [Results](#results)
- [Contributors](#contributors)

## Introduction

Pneumonia is a potentially life-threatening lung infection that requires prompt diagnosis. This project aims to create classifiers capable of accurately identifying pneumonia from chest X-ray images using various machine learning and deep learning techniques.

## Dataset

The dataset used in this project is the [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle. It consists of 5,863 X-ray images categorized into Pneumonia and Normal cases, with data divided into training, validation, and test sets.

## Project Structure

The project is divided into the following sections:

1. *Data Import*: Importing and organizing the dataset into training, validation, and test sets.
2. *Data Exploration*: Analyzing the dataset for class distribution, image quality, and other relevant features.
3. *Machine Learning Models*: Training and evaluating various machine learning models.
4. *Deep Learning Models*: Implementing a Convolutional Neural Network (CNN) for image classification.

## Machine Learning Models

The following machine learning algorithms were implemented:

- *Support Vector Machine (SVM)*
- *Decision Tree*
- *Random Forest*
- *AdaBoost*
- *XGBoost*

Each model was trained on the dataset, and hyperparameters were tuned for optimal performance. Key metrics such as accuracy, precision, recall, and F1-score were used to evaluate the models.

## Deep Learning Models

A Convolutional Neural Network (CNN) was developed using TensorFlow's Keras API. The network architecture includes:

- *Conv2D Layers*: For feature extraction from the images.
- *MaxPooling2D Layers*: For downsampling the image data.
- *Dropout Layers*: To prevent overfitting.
- *Dense Layers*: For the final classification.

Two types of data augmentations were applied during preprocessing: color jittering and layer augmentation.

## Results

The project achieved high accuracy in classifying pneumonia from normal cases across both machine learning and deep learning models. The Support Vector Machine (SVM) and AdaBoost models performed exceptionally well among the machine learning models, while the CNN demonstrated strong performance in the deep learning approach.

## Contributors
Benedetta Pacilli - benedetta.pacilli@studio.unibo.it <br>
Valentina Pieri - valentina.pieri5@studio.it
