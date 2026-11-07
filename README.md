# Online Shoppers Purchasing Intention Prediction

## A. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification models to predict whether an online shopping session will result in a purchase.

The project uses the Online Shoppers Purchasing Intention dataset to identify purchasing patterns based on user browsing behaviour, page interactions, session duration, visitor type, and other relevant attributes.

This is a binary classification problem where the target variable is `Revenue`:

- `True`: The visitor completed a purchase.
- `False`: The visitor did not complete a purchase.

## B. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

Dataset characteristics:

- Total records: 12,330
- Input features: 17
- Target variable: Revenue
- Problem type: Binary Classification

The dataset contains numerical and categorical features.

### Feature Description

| Feature | Description |
|---|---|
| Administrative | Number of administrative pages visited |
| Administrative_Duration | Time spent on administrative pages |
| Informational | Number of informational pages visited |
| Informational_Duration | Time spent on informational pages |
| ProductRelated | Number of product-related pages visited |
| ProductRelated_Duration | Time spent on product-related pages |
| BounceRates | Average bounce rate |
| ExitRates | Average exit rate |
| PageValues | Average page value |
| SpecialDay | Closeness of the visit to a special day |
| Month | Month of the visit |
| OperatingSystems | Operating system used |
| Browser | Browser used |
| Region | Geographic region |
| TrafficType | Traffic source type |
| VisitorType | Type of visitor |
| Weekend | Whether the visit occurred on a weekend |

## C. GitHub Repository Link

**GitHub Repository:** [Add your GitHub repository URL here]

## D. Models Used and Model Comparison

The following five machine learning classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation Metrics

The models were evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correctly classified predictions.
- **AUC:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures how many predicted purchases were actually purchases.
- **Recall:** Measures how many actual purchases were correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient, which evaluates classification quality while considering all four confusion matrix outcomes.

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8410 | 0.8932 | 0.4913 | 0.7435 | 0.5917 | 0.5145 |
| Decision Tree | 0.8487 | 0.7127 | 0.5117 | 0.5157 | 0.5137 | 0.4241 |
| K-Nearest Neighbors | 0.8767 | 0.7633 | 0.6822 | 0.3822 | 0.4899 | 0.4493 |
| Gaussian Naive Bayes | 0.2729 | 0.7334 | 0.1726 | 0.9738 | 0.2933 | 0.1289 |
| Support Vector Machine | 0.8654 | 0.8979 | 0.5477 | 0.7513 | 0.6336 | 0.5639 |

*The results are based on the held-out test dataset.*

## E. Observations on Model Performance

### 1. Logistic Regression

- Achieved an accuracy of 84.10%.
- Recorded a strong AUC of 0.8932.
- Achieved a recall of 74.35%, indicating that it identifies a substantial proportion of actual purchasing sessions.
- Its F1 Score of 0.5917 and MCC of 0.5145 indicate reasonably balanced classification performance.

### 2. Decision Tree

- Achieved an accuracy of 84.87%.
- Recorded an AUC of 0.7127.
- Precision and recall were relatively balanced at 0.5117 and 0.5157.
- Its overall performance was lower than Logistic Regression and SVM based on AUC, F1 Score, and MCC.

### 3. K-Nearest Neighbors (KNN)

- Achieved the highest accuracy of 87.67%.
- Recorded the highest precision of 0.6822.
- However, its recall was 0.3822, meaning it missed a considerable proportion of actual purchasing sessions.
- Its F1 Score was 0.4899, showing that high accuracy alone does not necessarily indicate balanced classification performance.

### 4. Gaussian Naive Bayes

- Achieved a recall of 97.38%, the highest among all evaluated models.
- However, its accuracy was only 27.29%, and its precision was 0.1726.
- The low precision indicates a large number of false-positive predictions.
- Its MCC of 0.1289 indicates weak overall classification performance.

### 5. Support Vector Machine (SVM)

- Achieved the highest AUC of 0.8979.
- Recorded the highest F1 Score of 0.6336 and MCC of 0.5639.
- Achieved a recall of 75.13%, demonstrating its ability to identify actual purchasing sessions.
- It provides the strongest overall balance between precision and recall among the evaluated models.

## F. Overall Winner

Based on the evaluation results, **Support Vector Machine (SVM)** is selected as the overall best-performing model.

It achieved:

- Accuracy: 86.54%
- AUC: 0.8979
- Precision: 0.5477
- Recall: 0.7513
- F1 Score: 0.6336
- MCC: 0.5639

Although K-Nearest Neighbors