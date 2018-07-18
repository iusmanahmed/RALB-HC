from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
data_file=pd.read_csv('application_estimator.csv')
trainx= data_file.iloc[:,  2:35]
trainy= data_file.iloc[:, 35]
X_train, X_test, y_train, y_test = train_test_split(trainx, trainy,train_size=0.90, test_size=0.10)

tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')
