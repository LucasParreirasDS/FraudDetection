import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, recall_score

# Carregando o dataset
df = pd.read_csv('data/raw/card_transdata.csv')

# Separando features usadas e target
x = df.drop(['fraud', 'distance_from_last_transaction', 'repeat_retailer'], axis=1)
y = df['fraud']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Padronizando escala dos dados
scaler = StandardScaler()
x_train_scl = scaler.fit_transform(x_train)
x_test_scl = scaler.fit_transform(x_test)



# Testando XGBClassifier
xgb = XGBClassifier(subsample=0.5) 
param_grid = {'learning_rate': [1.2, 1.5, 2],
              'max_depth': [6, 8, 10], 
              'gamma': [0]}

grid = GridSearchCV(xgb, param_grid=param_grid, cv=5, scoring=make_scorer(recall_score), verbose=3)

grid.fit(x_train_scl, y_train)

print('XGB Results')
print(grid.best_score_)
print(grid.best_params_) # {'gamma': 0, 'learning_rate': 1.2, 'max_depth': 10} | {'gamma': 0, 'learning_rate': 1.5, 'max_depth': 8}


# Testando regressão logística
lr = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 
              'solver': ['lbfgs', 'liblinear'], 
              'penalty': ['l1', 'l2']}

grid_lr = GridSearchCV(lr, param_grid=param_grid, cv=5, scoring=make_scorer(recall_score), verbose=3)

grid_lr.fit(x_train_scl, y_train)
print('LR')
print(grid_lr.best_score_)
print(grid_lr.best_params_)


'''
svc = SVC()
param_grid = { 'C':[1,100], 
              'kernel':['rbf','poly','sigmoid'], 
              'degree':[1,5], 
              'gamma': [1, 0.001]}
'''


svc = SVC(kernel='rbf', 
          C=75)    
svc.fit(x_train_scl ,y_train)
print(svc.score(x_test_scl, y_test)) # sigmoid = 0.90733 | rbf = 0.98999
y_pred = svc.predict(x_test_scl)

cmf = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(13,10))
sns.heatmap(cmf, cmap='coolwarm', annot=True, fmt='g')
plt.show()

'''grid = GridSearchCV(svc, param_grid=param_grid, cv=5, scoring=make_scorer(recall_score), verbose=3)
grid.fit(x_train_scl ,y_train)
print(grid.best_score_)
print(grid.best_params_)
'''