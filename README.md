# Digit Recognizer - Machine Learning

Welcome to my repository for the Digit Recognizer competition hosted on Kaggle. This project is part of my capstone project for the ICL Professional Certification Course in Machine Learning and Artificial Intelligence.

## Project Overview

This project aims to correctly identify digits from a dataset of tens of thousands of handwritten images. We use the MNIST ("Modified National Institute of Standards and Technology") dataset, which is a large database of handwritten digits that is commonly used for training various image processing systems.

## Competition Details

The Digit Recognizer competition on Kaggle is a practice contest to introduce machine learning beginners to the challenges of real-world data. More details about the competition can be found on the [official Kaggle competition page](https://www.kaggle.com/competitions/digit-recognizer).

## Repository Structure

Here's a brief overview of the repository:

- `train.csv`: The training set provided by the competition.
- `test.csv`: The test set for which you need to predict the labels.
- `sample_submission.csv`: A sample submission file in the correct format.
- `ICL_Digit_Recognizer.ipynb`: Jupyter notebook containing the code for the project.
- `model.h5`: Saved model weights after training.
- `requirements.txt`: A text file listing the project's dependencies.

## Models and Techniques Used

In this project, I employed a Convolutional Neural Network (CNN) to classify the handwritten digits. The model's architecture includes multiple convolutional and pooling layers, followed by dense layers for classification.

## Hyperparameter Tuning

I used the Keras Tuner library to find the optimal hyperparameters for the CNN. This process helped in improving the model's accuracy by experimenting with different configurations.

## Dependencies

This project requires Python 3.x, along with the following libraries:
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Keras Tuner

To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
