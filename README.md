# FinTech Fraud Detection ML

An end-to-end machine learning pipeline for detecting fraudulent digital transactions in a FinTech payment environment.

## 📌 Overview

Digital payment platforms face significant risks from fraudulent transactions, including financial losses, reduced customer trust, and reputational impact.

This project develops a machine learning-based fraud detection pipeline designed to identify potentially fraudulent transactions before completion.

The workflow covers the complete process from **synthetic data generation and preprocessing to exploratory analysis, model training, evaluation, and model selection**.

> **Note:** This project uses synthetic transaction data because real-world financial transaction data is sensitive and generally not publicly available.

## 🎯 Problem Statement

The objective is to build a machine learning solution capable of detecting fraudulent digital transactions using transaction, account, device, location, and network-related features.

The project focuses on identifying patterns associated with potentially fraudulent activity while addressing the highly imbalanced nature of fraud datasets.

## 📊 Dataset

The project generates a synthetic dataset containing:

* **200,000 transaction records**
* **2,000 fraudulent transactions**
* **1% fraud rate**
* **9 predictive features**
* **1 target variable**

### Features

| Feature                    | Description                                |
| -------------------------- | ------------------------------------------ |
| `transaction_amount`       | Transaction value                          |
| `transaction_time_seconds` | Transaction time represented in seconds    |
| `failed_attempts`          | Number of failed attempts                  |
| `account_age_days`         | Age of the account                         |
| `location_distance_km`     | Distance associated with the transaction   |
| `device_changes_30d`       | Device changes during the previous 30 days |
| `is_new_device`            | Indicates whether a new device was used    |
| `is_international`         | Indicates an international transaction     |
| `ip_address_changed`       | Indicates an IP address change             |
| `is_fraud`                 | Target variable: legitimate or fraudulent  |

## 🔄 Machine Learning Pipeline

```text
Synthetic Data Generation
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Feature Preparation
          ↓
Train/Test Split
          ↓
SMOTE Oversampling
          ↓
Model Training
          ↓
Model Evaluation
          ↓
F1-Score Based Model Selection
```

## 🧹 Data Cleaning

The dataset is checked for:

* Missing values
* Duplicate records
* Appropriate data types

The generated dataset contains no missing values or duplicate rows.

## 🔍 Exploratory Data Analysis

The project analyzes:

* Transaction amount distribution
* Fraudulent vs. legitimate transaction distribution
* Transaction amounts by fraud status
* Feature correlation
* Fraud patterns related to:

  * New devices
  * International transactions
  * IP address changes

These analyses help identify patterns that can support feature preparation and model selection.

## ⚖️ Handling Class Imbalance

Fraud detection typically involves a highly imbalanced target variable. In this project, only **1% of transactions are fraudulent**.

To address this imbalance, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied **only to the training data**. This helps prevent information from the test set from influencing the resampling process.

## 🤖 Models

The project evaluates three classification algorithms:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**

The models are compared using fraud-relevant classification metrics.

## 📈 Evaluation Metrics

The following metrics are used to evaluate model performance:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

The models are compared using F1 Score, and the model with the highest F1 Score is selected as the best-performing model within this experiment.

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **imbalanced-learn / SMOTE**
* **Google Colab / Jupyter Notebook**

## 📂 Project Structure

```text
fintech-fraud-detection-ml/
│
├── README.md
├── fraud_detection.ipynb
└── data/
    └── synthetic_transaction_data.csv
```

*The dataset file is optional because the notebook generates the synthetic dataset programmatically.*

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fintech-fraud-detection-ml.git
cd fintech-fraud-detection-ml
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### 3. Run the notebook

Open:

```text
fraud_detection.ipynb
```

The notebook generates the synthetic dataset and executes the complete fraud detection pipeline.

## 💡 Key Learning Outcomes

This project demonstrates practical application of:

* Fraud detection concepts in FinTech
* Synthetic financial data generation
* Data preprocessing and validation
* Exploratory data analysis
* Imbalanced classification
* SMOTE-based oversampling
* Multiple machine learning algorithms
* Classification model evaluation
* F1-score based model selection

## ⚠️ Disclaimer

This project is an educational and experimental implementation using **synthetic transaction data**. It is not trained on real Paytm customer or transaction data and should not be considered a production fraud detection system.

## 👩‍💻 Author

**Gaganpreet Kaur**
BBA – FinTech & AI
