import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

data=pd.read_csv("Churn_Modelling.csv")

#Categorizing our text columns, numeric columns, and target column(y)
text_cols=["Geography", "Gender"]
numeric_cols=["RowNumber", "CreditScore", "Age" , "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
y = data["Exited"]

#Identifying the text and numeric column of our feature data
x_text= data[text_cols]
x_numeric= data[numeric_cols]

#Vectorizing the text columns
x_vectorized_texts = []
for col in text_cols:
    vec = TfidfVectorizer()
    vectorized = vec.fit_transform(x_text[col])
    x_vectorized_texts.append(vectorized)

#Scaling the data in the numeric columns
scalar= StandardScaler()
x_numeric_scaled= scalar.fit_transform(x_numeric)

#Combining our vectorized test and sparsed scaled data into our feature data(x)
from scipy.sparse import csr_matrix
x = hstack(x_vectorized_texts + [csr_matrix(x_numeric_scaled)])

#Using train_test split to divide our data into train and test data
x_train, x_test, y_train, y_true = train_test_split(x, y, test_size= 0.3)

#Training the model
model= LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

#Predicting and printing the accuracy
y_prediction= model.predict(x_test)
accuracy= accuracy_score(y_true, y_prediction)
print(F"Accuracy: {accuracy}")
