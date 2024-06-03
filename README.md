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

## Usage

To replicate the findings and run the notebook:

1. Clone the repository.
2. Install the required dependencies.
3. Open the Digit_Recognizer.ipynb notebook in a Jupyter environment.
4. Run the cells sequentially to train the model and make predictions.

## Results

The accuracy of 99.31% on the validation set is excellent and implies that the model will likely perform well on similar real-world data. The slight gap between training accuracy (100%) and validation accuracy (99.31%) does indicate a minimal level of overfitting. Essentially, the model fits the training data slightly better than it fits new data.

## Contributing

Feel free to fork this project and make your own changes too. Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details: https://www.kaggle.com/competitions/digit-recognizer/data

## Contact Information

For any further inquiries, you can reach me at tatiana.massoco@gmail.com .

Thank you for your interest in this project!
