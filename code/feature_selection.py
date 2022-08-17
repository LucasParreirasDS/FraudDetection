import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

df = pd.read_csv('data/raw/card_transdata.csv')

x = df.copy()
y = x.pop('fraud')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
x_train_scl = scaler.fit_transform(x_train)
x_test_scl = scaler.fit_transform(x_test)

xgb = XGBClassifier()
xgb.fit(x_train_scl, y_train)

y_pred = xgb.predict(x_test_scl)

importances = pd.DataFrame(data={
    'Attribute': x_train.columns,
    'Importance': xgb.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(14,8))
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()

cmf = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cmf, cmap='Blues', annot=True, fmt='g')
plt.xlabel(' P R E V I S T O')
plt.ylabel('R E A L')
plt.show()