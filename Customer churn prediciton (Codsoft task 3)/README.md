# Customer Churn Prediction Model

## Project Description

This project implements a **Customer Churn Prediction Model** that uses machine learning to determine whether a customer is likely to leave a bank (churn). It processes both text and numeric data, transforming categorical variables (like gender and geography) and scaling numeric features (like credit score, balance, and salary). The model uses **Logistic Regression** to predict customer churn based on these inputs.

### Key Features:
- **Text Features:** Transform categorical text data using `TfidfVectorizer` to numerically represent features like `Gender` and `Geography`.
- **Numeric Features:** Apply `StandardScaler` to normalize numeric features, such as `CreditScore`, `Balance`, and `EstimatedSalary`.
- **Combining Features:** Merge the vectorized text features and the scaled numeric data using `scipy.sparse.hstack` to form the complete input feature set for the model.
- **Model:** Use a Logistic Regression model to predict if a customer will churn.

### Performance:
- **Accuracy:** The model achieves an accuracy of approximately **83%** on the test dataset, showing reasonable predictive ability for a basic implementation.

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

---

## File Structure

- `Churn_Modelling.csv`: Dataset containing customer information and churn status.
- `model.py`: Python script that includes data processing, model training, and evaluation logic.
- `requirements.txt`: Lists all necessary Python libraries.

---

## Usage

To run the churn prediction model, execute the script `model.py`:

```bash
python model.py
