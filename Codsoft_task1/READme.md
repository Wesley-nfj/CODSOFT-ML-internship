# 🎬 Movie Genre Predictor

## Project Description

This project implements a **Movie Genre Prediction** model that uses Natural Language Processing (NLP) to classify movies into genres based on their descriptions. By converting text data into numerical form and training a classifier, the model can predict a movie's genre using supervised learning techniques.

---

## 🔍 Key Features

- **Text Preprocessing**: Uses `TfidfVectorizer` to transform movie descriptions into numerical feature vectors.
- **Model Training**: A `LogisticRegression` classifier is trained on these features to learn the relationship between text and genres.
- **Evaluation**: The model's performance is measured using accuracy on a held-out test set.

---

## 🧰 Tools & Libraries Used

- `pandas` for data loading and handling
- `scikit-learn` for text vectorization, model training, and evaluation
- `TfidfVectorizer` for converting movie descriptions into vectors
- `LogisticRegression` for classification

---

## 🚀 How to Run

1. Clone this repository.
2. Ensure you have Python installed (preferably 3.8+).
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

