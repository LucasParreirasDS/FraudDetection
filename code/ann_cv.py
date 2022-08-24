import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Recall
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, recall_score, classification_report

def build_clf(unit):
    model = Sequential()
    
    model.add(Dense(n_features, activation='relu', input_shape=(n_features, )))
    model.add(Dense(unit, activation='relu'))
    model.add(Dense(unit, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss=BinaryCrossentropy(), optimizer='adam', metrics=Recall())
    
    return model


df = pd.read_csv('data/raw/card_transdata.csv')

x = df.drop(['fraud', 'distance_from_last_transaction', 'repeat_retailer'], axis=1).reset_index(drop=True)
y = df['fraud']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

n_features = x.shape[1]

param_grid = {'unit': [6, 12, 20]}

# Escalando os dados
scaler = StandardScaler()
x_train_scl = scaler.fit_transform(x_train)
x_test_scl = scaler.fit_transform(x_test)

'''x_train_scl = np.expand_dims(x_train, axis=-1)
x_test_scl = np.expand_dims(x_test, axis=-1)
'''
# Criando o modelo com KFold e GridSearch
kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

clf = KerasClassifier(build_fn=build_clf, epochs=20)

grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kfold, verbose=3, scoring=make_scorer(recall_score))


history = grid.fit(X=x_train_scl, y=y_train)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
sns.lineplot(data=history.history['accuracy'], ax=ax[0])
sns.lineplot(data=history.history['val_accuracy'], ax=ax[0])
ax[0].set_title('Acuracia por epoca')
ax[0].set_xlabel('épocas')
ax[0].set_ylabel('acurácia')
plt.legend(['treino', 'validacao'])
sns.lineplot(data=history.history['loss'], ax=ax[1])
sns.lineplot(data=history.history['val_loss'], ax=ax[1])
ax[1].set_title('Perda por epoca')
ax[1].set_xlabel('épocas')
ax[1].set_ylabel('perda')
plt.legend(['treino', 'validacao'])

plt.show()     

y_pred = grid.predict(x_test_scl)
cmf = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cmf, cmap='Blues', annot=True, fmt='g')
plt.xlabel(' P R E V I S T O')
plt.ylabel('R E A L')
plt.show()