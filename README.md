# Handwritten Digit Recognition with KNN

## Overview

This repository contains a Python notebook that demonstrates the use of the K-Nearest Neighbors (KNN) algorithm to recognize handwritten digits. The task is accomplished using the well-known MNIST dataset. This project provides a comprehensive walkthrough of the machine learning pipeline - from loading and exploring data to model training and evaluation.


For more details, please refer to the following links:

- [Kaggle Notebook](https://www.kaggle.com/your-kaggle-notebook-link)
- [Medium Article](https://www.medium.com/your-medium-article-link)


![01](https://github.com/abdullah1772/Classify-Handwritten-Digits-using-KNN/assets/88187437/5b22b06c-7d14-4ee5-98dd-0bef641c5c5a)



## Requirements

The code uses the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- sklearn
- keras

You can install these libraries using pip:

```
pip install numpy pandas matplotlib seaborn sklearn keras
```

## Dataset

The MNIST dataset, short for Modified National Institute of Standards and Technology, is a classic dataset used in computer vision and machine learning. It's a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

## Structure

The notebook is structured as follows:

1. **Loading the Dataset:** The MNIST dataset is loaded and divided into training and test sets.

2. **Exploratory Data Analysis (EDA):** Basic EDA is performed to get a better understanding of the data.

3. **Data Preprocessing:** The grayscale image data is normalized and reshaped to prepare it for the KNN model.

4. **Model Training:** A KNN model is trained using grid search to find the best combination of hyperparameters.

5. **Model Evaluation:** The trained model is evaluated on the test set using various metrics and techniques, including a classification report, a confusion matrix, and visualizations of the model's predictions for individual images.

## Usage

Clone this repository to your local machine and run the Jupyter notebook to see the code in action. If you want to use a different dataset, you can replace the data loading step with code to load your dataset.

## Contributing

Feel free to fork this repository and make your own changes. If you find any bugs or have any suggestions for improvement, please open an issue. Contributions are welcome!

## Acknowledgements

The MNIST dataset is publicly available and was originally created by Yann LeCun. It's hosted on many platforms. In this project, it's loaded using a utility function from the keras library.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
