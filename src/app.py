import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('data/interim/clean_titanic.csv')

X = df.drop(['Survived'],axis=1)
y = df['Survived']



X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y, random_state=15)

model = RandomForestClassifier(random_state=11, n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features='sqrt',max_depth=70, bootstrap=True)

model.fit(X_train,y_train)

filename = 'models/finalized_model.sav'

pickle.dump(model, open(filename,'wb'))