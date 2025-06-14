import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
 

train_df = pd.read_csv("fraudTrain.csv")
test_df = pd.read_csv("fraudTest.csv")


#categorizing columns into text columns, numeric columns and identifying the target column
text_cols = ["merchant", "category", "first", "last", "street", "city", "state", "job"]
numeric_cols = ["amt", "zip", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
target= "is_fraud"

#Extracting the text and numeric feature columns from the train data, and also identifying the target of the train data
x_train_text= train_df[text_cols]
x_train_numeric= train_df[numeric_cols]
y_train= train_df[target]

#Extracting the text and numeric feature columns from the test data, and also identifying the target of the test data
x_test_text= test_df[text_cols]
x_test_numeric= test_df[numeric_cols]
y_true= test_df[target]

#Vectoring each text column
vectorized_train_texts = []
vectorized_test_texts = []

for col in text_cols:
    vec = TfidfVectorizer()
    train_vec = vec.fit_transform(x_train_text[col])
    test_vec = vec.transform(x_test_text[col])
    vectorized_train_texts.append(train_vec)
    vectorized_test_texts.append(test_vec)


#Scaling the numeric columns
scalar= StandardScaler()
x_train_num_scaled= scalar.fit_transform(x_train_numeric)
x_test_num_scaled= scalar.transform(x_test_numeric)

# Convert numeric data to sparse matrix
x_train_num_sparse = csr_matrix(x_train_num_scaled)
x_test_num_sparse = csr_matrix(x_test_num_scaled)

# Combine vectorized text and scaled data into feature of train and test data
x_train = hstack(vectorized_train_texts + [x_train_num_sparse])  
x_test = hstack(vectorized_test_texts + [x_test_num_sparse]) 


#Training the model
model= LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

#prediting and printing the accuracy score
y_prediction= model.predict(x_test)

accuracy=accuracy_score(y_true, y_prediction)
print(f"accuracy: {accuracy}")

#using matplotlib for data visualisation 
class_counts= train_df["is_fraud"].value_counts()
plt.bar(class_counts.index, class_counts.values, color=['#4CAF50', '#F44336'])
plt.xlabel("Is Fraud")
plt.ylabel("Number of Transactions")
plt.title("Class Distribution")
plt.tight_layout()
plt.show()

