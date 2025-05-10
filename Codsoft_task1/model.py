import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score

train_df= pd.read_csv("train_data.txt",sep= ":::", engine='python', header= None)
train_df.columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
test_df= pd.read_csv("test_data.txt", sep=":::", engine='python', header= None)
test_df.columns = ['ID', 'TITLE', 'DESCRIPTION']
solution_df= pd.read_csv("test_data_solution.txt", sep=":::", engine='python', header= None)
solution_df.columns = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']

#Identifying feature data
x_train=train_df["DESCRIPTION"]
y_train=train_df["GENRE"]

#vectorizing x_train using TF-IDF
vectorizer=TfidfVectorizer()
x_train= vectorizer.fit_transform(x_train)

#Training the model
model=LogisticRegression(max_iter= 1000)
model.fit(x_train, y_train)

#Identifying and vectorizing target data
x_test=test_df["DESCRIPTION"]
x_test= vectorizer.transform(x_test)
y_solution=solution_df["GENRE"]

#prediting and priting the accuracy score
y_prediction= model.predict(x_test)

accuracy= accuracy_score(y_solution, y_prediction,)
print(f"accuracy: {accuracy}")
