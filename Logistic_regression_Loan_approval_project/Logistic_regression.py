import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("loan_approval_data.csv")


data["Credit_History"] = data["Credit_History"].fillna(data["Credit_History"].mode()[0])
data["Dependents"] = data["Dependents"].fillna(data["Dependents"].mode()[0])
data["Gender"] = data["Gender"].fillna(data["Gender"].mode()[0])
data["Self_Employed"] = data["Self_Employed"].fillna(data["Self_Employed"].mode()[0])
data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())
data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0])

data.drop(["Loan_ID"] , axis=1 , inplace= True)

data = pd.get_dummies(data, columns = ["Gender","Education",'Self_Employed','Married',"Property_Area"])
data.drop(["Gender_Female" ,"Education_Not Graduate",'Self_Employed_No','Married_No','Property_Area_Rural'], axis = 1 , inplace = True )

data['Dependents'] = data['Dependents'].replace('3+' ,3)
data['Dependents']= data['Dependents'].astype(int)

x = data.drop(["Loan_Status"] , axis = 1)
y = data["Loan_Status"]




from sklearn.model_selection import train_test_split

x_train , x_test ,y_train , y_test = train_test_split(x , y , train_size= 0.30 , random_state= 42)

from sklearn.linear_model import LogisticRegression
Model = LogisticRegression(solver='saga')

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

from sklearn.model_selection import GridSearchCV
parameter = {
    "penalty": ['l1','l2','elasticnet'],
    'C': [.001,.01,1,3,5,10,20,40,50,60,100],
    'max_iter': [10,20,30,40,50,70,90,100],
    'l1_ratio': [.1,.2,.3,.4,.5,.6,.7,.8,.9]
}
classifier = GridSearchCV(Model,param_grid=parameter,scoring='accuracy' , cv = 5)
classifier.fit(x_train,y_train)

print(classifier.best_params_)
print(classifier.best_score_)

y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score , classification_report

print(accuracy_score(y_pred , y_test))
print(classification_report(y_pred , y_test))

selected_cols = ['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Loan_Status']
sns.pairplot(data[selected_cols], hue="Loan_Status", height=3.5)
plt.show()

sns.countplot(x='Loan_Status', data=data, palette='Set2')
plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()

sns.countplot(x="Credit_History" , hue='Loan_Status' , data=data , palette='Set1')
plt.xlabel('Credit History(0: Bad , 1: Good)')
plt.title('Loan status by credit history')
plt.ylabel('Count')
plt.show()




