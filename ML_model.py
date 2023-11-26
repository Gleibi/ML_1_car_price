import joblib
import pandas as pd
import numpy as np
import random
from sklearn import impute
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)

df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')
df_test = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_test.csv')

df_train = df_train.drop_duplicates(subset=['name','year','fuel','seller_type','km_driven','owner','mileage','engine','max_power','torque'],keep='first')
df_train = df_train.reset_index(drop=True)

df_train[['torque', 'max_torque_rpm']] = df_train['torque'].str.extract('(\d+\.\d+)Nm@ (\d+)rpm').astype(float)
df_test[['torque', 'max_torque_rpm']] = df_test['torque'].str.extract('(\d+\.\d+)Nm@ (\d+)rpm').astype(float)

split_mileage = df_train['mileage'].str.split(' ', n=1, expand=True)
df_train['mileage'] = pd.to_numeric(split_mileage[0], errors='coerce').astype(float)

split_engine = df_train['engine'].str.split(' ', n=1, expand=True)
df_train['engine'] = pd.to_numeric(split_engine[0], errors='coerce').astype(float)

split_max_power = df_train['max_power'].str.split(' ', n=1, expand=True)
df_train['max_power'] = pd.to_numeric(split_max_power[0], errors='coerce').astype(float)

split_mileage = df_test['mileage'].str.split(' ', n=1, expand=True)
df_test['mileage'] = pd.to_numeric(split_mileage[0], errors='coerce').astype(float)

split_engine = df_test['engine'].str.split(' ', n=1, expand=True)
df_test['engine'] = pd.to_numeric(split_engine[0], errors='coerce').astype(float)

split_max_power = df_test['max_power'].str.split(' ', n=1, expand=True)
df_test['max_power'] = pd.to_numeric(split_max_power[0], errors='coerce').astype(float)

cat_features_mask = (df_train.dtypes == "object").values

df_real_train = df_train[df_train.columns[~cat_features_mask]]
mis_replacer = impute.SimpleImputer(strategy="mean")
df_no_mis_real_train = pd.DataFrame(data=mis_replacer.fit_transform(df_real_train), columns=df_real_train.columns)

df_cat_train = df_train[df_train.columns[cat_features_mask]].fillna("")
df_cat_train.reset_index(drop=True, inplace=True)

df_no_mis_train = pd.concat([df_cat_train,df_no_mis_real_train], axis=1)

df_real_test = df_test[df_test.columns[~cat_features_mask]]
mis_replacer = impute.SimpleImputer(strategy="mean")
df_no_mis_real_test = pd.DataFrame(data=mis_replacer.fit_transform(df_real_test), columns=df_real_test.columns)

# для категориальных - пустыми строками
df_cat_test = df_test[df_test.columns[cat_features_mask]].fillna("")
df_cat_test.reset_index(drop=True, inplace=True)

df_no_mis_test = pd.concat([df_cat_test,df_no_mis_real_test], axis=1)

df_no_mis_train['engine'] = df_no_mis_train['engine'].astype(int)
df_no_mis_train['seats'] = df_no_mis_train['seats'].astype(int)

df_no_mis_test['engine'] = df_no_mis_test['engine'].astype(int)
df_no_mis_test['seats'] = df_no_mis_test['seats'].astype(int)

y_train = df_no_mis_train['selling_price']
X_train = df_no_mis_train.drop(['selling_price', 'name', 'fuel', 'seller_type', 'transmission', 'owner', 'torque','engine'], axis=1)

y_test = df_no_mis_test['selling_price']
X_test = df_no_mis_test.drop(['selling_price', 'name', 'fuel', 'seller_type', 'transmission', 'owner', 'torque','engine'], axis=1)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.00001, l1_ratio=0.1)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

import joblib

joblib.dump(model, 'ElasticNet_model.pkl')

scaler.fit(pd.concat([df_real_train, df_real_test], axis=0))
mis_replacer.fit(pd.concat([df_real_train, df_real_test], axis=0))


joblib.dump(scaler, 'scaler.pkl')
joblib.dump(mis_replacer, 'imputer.pkl')