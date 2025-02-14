# gaussian-naive-bayes-implementation
# Naïve Bayes Classifier for Iris Dataset

## Overview
This project implements a **Gaussian Naïve Bayes classifier from scratch** to classify the Iris dataset. The classifier handles continuous-valued input attributes using the Gaussian distribution.

## Features
- Loads the **Iris dataset** from `sklearn.datasets`.
- Splits the dataset into **training (80%)** and **testing (20%)** sets.
- Computes **mean and standard deviation** for each feature per class.
- Uses the **Gaussian probability distribution** to calculate likelihoods.
- Applies **Bayes' Theorem** to compute posterior probabilities.
- Evaluates performance using **accuracy, confusion matrix, and classification report**.
- Provides **PCA-based visualization** of the decision boundaries.

## Installation
### Prerequisites
Ensure you have Python installed along with the necessary dependencies.
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage
### Running the Classifier
Clone this repository and execute the script:
```bash
git clone https://github.com/yourusername/naive-bayes-iris.git
cd naive-bayes-iris
python naive_bayes_iris.py
```

### Expected Output
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report**
- **PCA Visualization of the Dataset**

## Implementation Details
- Computes **mean** and **standard deviation** for each feature within each class.
- Uses the **Gaussian probability density function** to determine feature likelihood.
- Predicts class labels by computing **posterior probabilities**.

## Example Output
```
Accuracy: 95.67%
Confusion Matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
Classification Report:
               precision    recall  f1-score   support

    setosa       1.00      1.00      1.00        10
versicolor       0.90      0.90      0.90        10
 virginica       1.00      0.91      0.95        11
```

## Visualization
The PCA plot provides a 2D representation of the Iris dataset:

![PCA Visualization](pca_plot.png)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributions
Contributions are welcome! Feel free to open issues and submit pull requests.

## Author
[D.Anu Kumari](https://github.com/942004)

