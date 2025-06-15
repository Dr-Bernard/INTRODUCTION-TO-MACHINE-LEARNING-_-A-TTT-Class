# INTRODUCTION-TO-MACHINE-LEARNING-_-A-TTT-Class

# Laptop Price Prediction using Supervised Learning (Regression)

## Project Overview

This Jupyter Notebook, `Share HCIA-AI Script.ipynb`, serves as an educational guide and practical demonstration focusing on **Supervised Learning**, specifically **Regression**, to predict laptop prices. It meticulously explains fundamental machine learning concepts related to supervised learning before delving into a hands-on experiment of building and evaluating regression models.

## Table of Contents

1.  [Introduction to Supervised Learning & Regression](#introduction-to-supervised-learning--regression)
2.  [Experiment: Laptop Price Prediction (A Regression Task)](#experiment-laptop-price-prediction-a-regression-task)
    * [Data Loading and Exploratory Data Analysis (EDA)](#data-loading-and-exploratory-data-analysis-eda)
    * [Regression Model Training and Evaluation](#regression-model-training-and-evaluation)
3.  [Prerequisites](#prerequisites)
4.  [How to Run](#how-to-run)
5.  [Results](#results)
6.  [Author](#author)

## Introduction to Supervised Learning & Regression

The notebook begins by providing a clear foundation in key machine learning concepts, with a strong emphasis on **Supervised Learning**:

* **Definition of Machine Learning**: Explains how computers learn from data to make decisions.
* **Types of Machine Learning**: Differentiates between Supervised Learning and Unsupervised Learning, highlighting that **Supervised Learning** is the primary focus of this notebook due to its use of labeled datasets.
* **Supervised Learning and Regression**: Delves deeper into **Supervised Learning**, specifically introducing **Regression**. Regression is defined as a powerful supervised learning method used to model and predict a continuous target variable based on one or more independent variables. The laptop price prediction task is a classic example of a regression problem.
* **Linear Regression**: Provides a detailed explanation of Linear Regression, including both Simple Linear Regression (with one independent variable) and Multiple Linear Regression (with multiple independent variables). The fundamental formula `y = a + bx` for Simple Linear Regression is introduced and explained.

## Experiment: Laptop Price Prediction (A Regression Task)

This section outlines the practical steps taken to build and evaluate **regression models** for predicting laptop prices.

### Data Loading and Exploratory Data Analysis (EDA)

The initial phase involves understanding the dataset, which is crucial for any supervised learning task:

* **Dataset**: The project exclusively uses `laptop_price.csv` for training and testing the regression models.
* **Loading Data**: Utilizes the `pandas` library to efficiently load the dataset into a DataFrame.
* **Initial Data Inspection**: This crucial EDA step helps in understanding the features that will be used to predict the continuous 'Price_euros' target variable:
    * Displaying the first few rows (`.head()`).
    * Checking dataset dimensions (`.shape`).
    * Summarizing data types and non-null values (`.info()`).
    * Counting unique values per column (`.nunique()`).
    * Generating descriptive statistics for numerical columns (`.describe()`).
    * Verifying for duplicate rows and missing values (`.duplicated().sum()`, `.isnull().sum()`).
    * Inspecting unique values for categorical features like 'Company' and 'Product' to understand potential encoding needs for regression.

### Regression Model Training and Evaluation

The experiment proceeds with training and evaluating different **regression models**:

* **Linear Regression**:
    * A linear regression model, a fundamental supervised learning algorithm for continuous prediction, is trained on the preprocessed dataset.
    * Predictions are then generated on the test set using this trained model.
* **Random Forest Regressor**:
    * A Random Forest Regressor model, a more advanced ensemble supervised learning technique often yielding better performance, is also implemented and trained.
    * Predictions are generated using this robust model.
* **Regression Model Evaluation**:
    * Both regression models are rigorously evaluated using common regression-specific metrics from `sklearn.metrics`:
        * **R-squared (R2 Score)**: A key metric to assess how well the model's predictions align with the actual continuous target values, indicating the proportion of variance in the dependent variable that is predictable from the independent variables.
        * **Mean Squared Error (MSE)**: Measures the average of the squares of the errors, providing a measure of the average magnitude of the errors.

## Prerequisites

To run this notebook and explore the supervised learning regression models, you will need:

* Python 3.x
* Jupyter Notebook
* The following Python libraries:
    * `pandas`
    * `numpy`
    * `scikit-learn` (sklearn)

You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn jupyter
