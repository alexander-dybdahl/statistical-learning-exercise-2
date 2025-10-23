# Wine Quality Classification Using Statistical Learning Methods

## Overview

This project investigates the classification of Portuguese Vinho Verde wine quality using multiple statistical learning approaches. The study compares Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), Logistic Regression, tree-based methods, Random Forest, and Support Vector Machines (SVM) for binary classification of wine quality. Wine samples with quality scores of 7 or higher are classified as "good" (1), while those below 7 are classified as "bad" (0). The analysis uses the red wine quality dataset from Kaggle containing 1599 observations with 11 physicochemical features. After systematic comparison across multiple performance metrics, Random Forest emerged as the optimal classifier with 91.25% accuracy, the lowest p-value (0.0002064), and highest AUC (0.9024).

## Mathematical Formulation

### Binary Classification Problem

Let $\mathbf{x} \in \mathbb{R}^{11}$ represent the feature vector of physicochemical properties:

$$\mathbf{x} = [x_1, x_2, \ldots, x_{11}]^T$$

where each feature corresponds to: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol content.

The target variable $y \in \{0, 1\}$ is defined as:

$$y = \begin{cases}
1 & \text{if quality} \geq 7 \text{ (good wine)} \\
0 & \text{if quality} < 7 \text{ (bad wine)}
\end{cases}$$

### Classification Methods

#### Linear Discriminant Analysis (LDA)
The discriminant function assumes equal covariance matrices:

$$\delta_k(\mathbf{x}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2} \boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log \pi_k$$

where $\boldsymbol{\mu}_k$ is the class mean, $\boldsymbol{\Sigma}$ is the pooled covariance matrix, and $\pi_k$ is the prior probability.

#### Quadratic Discriminant Analysis (QDA)
Allows different covariance matrices per class:

$$\delta_k(\mathbf{x}) = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) - \frac{1}{2} \log |\boldsymbol{\Sigma}_k| + \log \pi_k$$

#### Logistic Regression
Models the log-odds using the sigmoid function:

$$P(Y = 1 | \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \boldsymbol{\beta}^T \mathbf{x})}}$$

#### Support Vector Machine (SVM)
Finds the optimal hyperplane with maximum margin using the radial basis function kernel:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

The decision function is:

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$$

## Method and Algorithms

The analysis implements a comprehensive comparison framework with 70-30 train-test split and fixed random seed for reproducibility. Each method employs specific optimization strategies: LDA and QDA use maximum likelihood estimation with pooled and class-specific covariance matrices respectively. Logistic regression optimizes the log-likelihood function using iterative reweighted least squares. Tree-based methods compare deviance and Gini splitting criteria with cross-validation pruning to prevent overfitting. Random Forest uses bootstrap aggregating with 500 trees and optimal mtry parameter selection. SVM employs 10-fold cross-validation for hyperparameter tuning with cost values ranging from 0.1 to 1000. Model evaluation uses accuracy, statistical significance (p-values), and Area Under the ROC Curve (AUC) as performance metrics.

## Repository Structure

```
├── Comp2.rmd          # Main R Markdown analysis document
├── Comp2.html         # Compiled HTML report
├── README.md          # Project documentation
└── requirements.txt   # R package dependencies
```

## Quick Start

### Environment Setup

Install R and required packages:

```r
# Install required packages
install.packages(c("knitr", "rmarkdown", "GGally", "dplyr", "e1071", 
                   "caret", "ggplot2", "gridExtra", "pROC", "randomForest", 
                   "MASS", "tree"))
```

### Minimal Run Command

```r
# Render the R Markdown document
rmarkdown::render("Comp2.rmd")
```

## Reproducing Results

### Data Requirements

Download the wine quality dataset:
- Source: [Red Wine Quality Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- Place the `winequality-red.csv` file in a `data/` directory

### Reproduction Commands

```r
# Set working directory
setwd("path/to/statistical-learning-exercise-2")

# Set random seed for reproducibility
set.seed(1)

# Render the complete analysis
rmarkdown::render("Comp2.rmd", output_format = "html_document")

# Alternative: run interactively
source("Comp2.rmd")  # Note: may require code extraction
```

### Configuration

The analysis uses the following key parameters:
- Random seed: 1
- Train-test split: 70-30
- Random Forest: 500 trees, mtry=3 for random forest, mtry=11 for bagging
- SVM: Radial kernel, 10-fold cross-validation
- Quality threshold: 7 (wines ≥7 classified as good)