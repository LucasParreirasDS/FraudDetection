import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, recall_score, classification_report

df = pd.read_csv('data/raw/card_transdata.csv')

x = df.drop(['fraud', 'distance_from_last_transaction', 'repeat_retailer'], axis=1)
y = df['fraud']

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=True, stratify=y)

scaler = StandardScaler()
x_scl = scaler.fit_transform(x)
#x_test_scl = scaler.fit_transform(x_test)

xgb = XGBClassifier()
kfold = StratifiedKFold(n_splits=5)

lst_accu_stratified = []
for train_idx, test_idx in kfold.split(x_scl, y):
    x_train_fold, x_test_fold = x_scl[train_idx], x_scl[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    xgb.fit(x_train_fold, y_train_fold)
    lst_accu_stratified.append(xgb.score(x_test_fold, y_test_fold))
    
y_pred = xgb.predict(x_test_fold)
print(f'Acurácia media: {lst_accu_stratified.mean()}')
print(classification_report(y_test_fold, y_pred))

cmf = confusion_matrix(y_test_fold, y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cmf, cmap='Blues', annot=True, fmt='g')
plt.xlabel(' P R E V I S T O')
plt.ylabel('R E A L')
plt.show()