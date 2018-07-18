from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import RandomOverSampler,SMOTE
data_file=pd.read_csv('data.csv')
trainx= data_file.iloc[:, 2:26]
trainy= data_file.iloc[:, 26]
print len(trainx)
ros = SMOTE(k_neighbors=4)
trainx,trainy= ros.fit_sample(trainx, trainy)
print len(trainx)

X_train, X_test, y_train, y_test = train_test_split(trainx, trainy,
                                                    train_size=0.90, test_size=0.10)

tpot = TPOTClassifier(generations=10, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')