import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


df = pd.read_csv('data/interim/clean_titanic.csv')

X = df.drop(['Survived'],axis=1)
y = df['Survived']



X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y, random_state=15)

# Random Forest

rf_model = RandomForestClassifier(random_state=11, n_estimators=1000, min_samples_split=2, min_samples_leaf=2, max_features='sqrt',max_depth=70, bootstrap=True)

rf_model.fit(X_train,y_train)

filename = 'models/rf_finalized_model.sav'

pickle.dump(rf_model, open(filename,'wb'))

#XGBoost

xgb_cl2 = xgb.XGBClassifier(colsample_bytree = 0.5, gamma= 1, learning_rate= 0.05, max_depth= 7, reg_lambda= 1, scale_pos_weight= 1, subsample= 0.8, objective="binary:logistic")
xgb_cl2.fit(X_train, y_train)

filename = 'models/xgb_finalized_model.sav'

pickle.dump(xgb_cl2, open(filename,'wb'))