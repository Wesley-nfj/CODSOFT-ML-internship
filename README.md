# Machine Learning Projects Portfolio

Welcome to my Machine Learning Projects Repository! This portfolio includes three beginner-friendly yet practical ML projects that demonstrate the use of supervised learning techniques on real-world datasets. Each project involves data preprocessing, model training, and evaluation using **Logistic Regression** and other essential tools.

---

## 📌 Projects Overview

### 1. 🎬 Movie Genre Prediction
- **Goal**: Predict the genre of a movie based on its description or text-based features.
- **Approach**:
  - Used `TfidfVectorizer` to convert movie descriptions into numerical feature vectors.
  - Trained a classification model to predict genres from text data.
- **Highlights**:
  - Great for NLP beginners.
  - Hands-on use of TF-IDF and Logistic Regression.

---

### 2. 💳 Credit Card Fraud Detection
- **Goal**: Detect fraudulent transactions from customer credit card transaction data.
- **Approach**:
  - Combined `TfidfVectorizer` for categorical text features (like merchant, job, etc.) and `StandardScaler` for numeric features.
  - Used `scipy.sparse.hstack` to merge both types into a unified feature set.
  - Trained a Logistic Regression model.
- **Performance**: Achieved approximately **99.5% accuracy**.
- **Highlights**:
  - Demonstrates hybrid data handling (text + numeric).
  - Realistic fraud detection scenario.

---

### 3. 👥 Customer Churn Prediction
- **Goal**: Predict whether a customer will leave a bank (churn) based on account and demographic data.
- **Approach**:
  - Text features like geography and gender were vectorized.
  - Numeric features were scaled using `StandardScaler`.
  - Combined features and trained a Logistic Regression model.
- **Performance**: Achieved over **83% accuracy**.
- **Highlights**:
  - Focuses on customer behavior modeling.
  - Illustrates simple preprocessing and classification techniques.

---

## 🧰 Tools & Libraries Used

- `pandas` for data handling
- `scikit-learn` for preprocessing, modeling, and evaluation
- `scipy` for handling sparse matrices
- `TfidfVectorizer` for text feature extraction
- `StandardScaler` for feature scaling
- `LogisticRegression` for classification

---

## 🚀 How to Run

1. Clone this repository.
2. Navigate to the project directory you want to explore.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
---

## Summary
These projects collectively demonstrate key aspects of machine learning workflows, including:
 'Data Preprocessing': Handling both text and numeric data, dealing with missing values, scaling, and vectorizing.
 'Feature Engineering': Converting categorical data into numerical format and combining multiple feature types.
 'Model Training & Evaluation': Using Logistic Regression to train models and assess performance.

By building and evaluating these models, I gained hands-on experience in text classification, fraud detection,
and customer behavior analysis, providing me with a strong foundation in machine learning. These projects are
ideal for those starting their journey in data science and machine learning, offering practical exposure to 
real-world datasets and problem-solving techniques.
