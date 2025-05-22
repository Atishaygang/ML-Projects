import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

data = pd.read_csv("train.csv")

data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0] )
data.drop(['Age' ,'Cabin'] , axis=1 , inplace= True)
data.drop(["Name" , "Ticket", 'PassengerId'] , axis = 1 , inplace=True)

data = pd.get_dummies(data , columns = ['Sex','Embarked']  )
data.drop(['Sex_female' ,'Embarked_Q'] , axis=1 , inplace=True)
print(data.isnull().sum())
x = data.drop(['Survived'] , axis = 1)
y = data['Survived']

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.30 , random_state=42)

from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
x_train = Scaler.fit_transform(x_train)
x_test = Scaler.transform(x_test)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

from sklearn.model_selection import cross_val_score
mse = cross_val_score(model , x_train , y_train , scoring="neg_mean_squared_error", cv=10)
r2 = cross_val_score(model , x_train , y_train , scoring="r2" , cv = 10)
print(np.mean(r2))
print(np.mean(mse))

prd_val = model.predict(x_test)

print(prd_val)
sns.displot(prd_val - y_test , kind='kde')
plt.show()





