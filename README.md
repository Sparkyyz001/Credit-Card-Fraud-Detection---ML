# 💳 Credit Card Fraud Detection using Machine Learning

This project implements a **Random Forest Classifier** to detect fraudulent credit card transactions. The system analyzes transaction patterns to distinguish between normal and suspicious activity in highly imbalanced datasets.

## 🎯 Project Goal
The primary objective is to build a robust model that helps financial institutions flag fraud early, reducing potential risks while maintaining a seamless experience for valid users.

## 🚀 Key Features
* **Exploratory Data Analysis (EDA):** Visualizing class imbalance and feature correlations.
* **Data Preprocessing:** Handling skewed data and splitting into training/testing sets.
* **Random Forest Model:** A powerful ensemble learning method for high-accuracy classification.
* **Performance Metrics:** Detailed evaluation using Precision, Recall, F1-Score, and Confusion Matrix.

## 🛠 Tech Stack
* **Language:** Python 3.x
* **Libraries:** * `pandas` & `numpy` (Data manipulation)
  * `matplotlib` & `seaborn` (Visualization)
  * `scikit-learn` (Machine Learning)

## 📊 Results & Evaluation
The model achieves a high balance between detecting fraud and minimizing false alarms.

### Confusion Matrix
Below is the performance breakdown on the test set:
- **True Negatives:** Successfully identified valid transactions.
- **True Positives:** Successfully caught fraud cases.
- **False Positives:** Minimized "false alarms" for real users.

    You can download the dataset from here
          https://media.geeksforgeeks.org/wp-content/uploads/20240904104950/creditcard.csv

## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/fraud-detection.git](https://github.com/your-username/fraud-detection.git)