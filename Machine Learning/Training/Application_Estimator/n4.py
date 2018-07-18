import numpy 
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from sklearn import cross_validation 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

data_file=pandas.read_csv('application_estimator.csv')
Input =numpy.array(data_file.iloc[:, 2:34])
Output= numpy.array(data_file.iloc[:, 34])
#print Output

import matplotlib.pyplot as plt


trainx, testx, trainy, testy = cross_validation.train_test_split(Input, Output, train_size=0.70, random_state=20000)

'''
#n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls'
est = GradientBoostingRegressor().fit(trainx, trainy)
print "mean_squared_error",mean_squared_error(testy, est.predict(testx)) 

import matplotlib.pyplot as plt
prediction=est.predict(testx)


print "percentage Error"

z=numpy.mean(numpy.abs(((testy) - (prediction)) / (testy))) * 100
print "Accuracy:",100-z

model= make_pipeline(
    StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=8, p=2, weights="distance")),
    GradientBoostingRegressor()
)
model = make_pipeline(
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, min_samples_split=15)),
    LassoLarsCV(normalize=False)
)

model=GradientBoostingRegressor()
est = model.fit(trainx, trainy)
print "mean_squared_error",mean_squared_error(testy, est.predict(testx)) 



'''
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor

model = make_pipeline(
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=5, min_samples_leaf=12, min_samples_split=16)),
    XGBRegressor(learning_rate=0.5, max_depth=10, min_child_weight=14, n_estimators=100, nthread=1, subsample=0.95)
)

from sklearn.externals import joblib
model.fit(Input, Output)
#print("actual ",model.feature_importances_)
joblib.dump(model, "application_estimator.joblib.pkl", compress=9)

est = model.fit(trainx, trainy)
print "mean_squared_error",mean_squared_error(testy, est.predict(testx)) 
from sklearn.metrics import r2_score
print r2_score(testy, est.predict(testx))  

prediction=est.predict(testx)

from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
"""
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StandardScaler(),
    GradientBoostingRegressor()
)
scores = cross_val_score(model, Input, Output, cv=5,scoring='neg_mean_squared_error')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print scores
"""
#plot_learning_curves(trainx,trainy, testx, testy, model,scoring='mean_squared_error',print_model=False)
#plt.show()

#http://rasbt.github.io/mlxtend/user_guide/plotting/plot_learning_curves/
"""
for x in range(0,len(prediction)):
	print "Actual:",round(testy[x], 3) , "Predicted:",round(prediction[x], 3) 

"""
dataer=round(mean_squared_error(testy, est.predict(testx)),3)

dataer2=round(r2_score(testy, est.predict(testx)) ,3)
plt.plot(prediction,'g--',label="Prediction")
plt.plot(testy,'r--',label="Actual")
plt.plot(mean_squared_error(testy, est.predict(testx)),label="Mean Sqaure Error = "+str(dataer))
plt.plot(mean_squared_error(testy, est.predict(testx)),label="R2 Score = "+str(dataer2))
plt.xlabel('OpenCL Kernal Functions',fontsize=16)
plt.ylabel('Relative speedup',fontsize=16)
#plt.title("RElative ")
plt.legend()
plt.show()



'''
est = GradientBoostingRegressor().fit(Input, Output)
print(est.feature_importances_)
joblib.dump(est, "speedup.joblib.pkl", compress=9)

'''