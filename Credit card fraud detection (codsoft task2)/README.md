# Credit Card Fraud Detection Model

## Project Description

This project implements a **Credit Card Fraud Detection Model** that uses machine learning techniques and Natural Language Processing (NLP) to identify fraudulent transactions from a dataset containing transaction details. It processes both text and numeric data, transforming categorical variables (like merchant names, city, etc.) and scaling numeric values (like transaction amount and coordinates). The model uses **Logistic Regression** to classify transactions as fraudulent or non-fraudulent.
To help understand the data, the project includes simple and clear visualizations of class distributions using matplotlib. These plots use colored bars and subtle gridlines to effectively highlight the imbalance between fraudulent and legitimate transactions.

### Key Features:
- **Text Features:** Transform categorical text data using `TfidfVectorizer` to create numerical representations for columns like merchant names, categories, job titles, and more.
- **Numeric Features:** Apply `StandardScaler` to normalize numeric data, ensuring all features contribute equally to the model's predictions.
- **Model:** Train a Logistic Regression model to predict fraudulent transactions based on processed data.
- **Visualization:** Includes simple and clear bar charts using matplotlib with colored bars and subtle gridlines to show class distribution for better interpretability.

### Performance:
- **Accuracy:** The model achieves an accuracy of approximately **99.56%** on the test dataset, demonstrating its strong ability to detect fraud.

---

## Setup and Installation

1. Clone the repository

2. Navigate to the project folder:

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Ensure that you have the following Python packages installed:
    - `pandas`
    - `scikit-learn`
    - `scipy`
    - `matplotlib`

---

## File Structure

- `fraudTrain.csv`: The training dataset containing transaction information.
- `fraudTest.csv`: The test dataset used to evaluate the model.
- `model.py`: Python script that contains the logic for feature processing, model training, and evaluation.
- `requirements.txt`: List of Python dependencies.

---

## Usage

To run the model, execute the script `model.py`:

```bash
python model.py
