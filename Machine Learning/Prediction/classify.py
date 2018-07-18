import pandas
from scipy.stats import rankdata
import numpy
from sklearn.externals import joblib
from scipy.stats import rankdata
import os

model1 = joblib.load("device_suitability.joblib.pkl")
model2 = joblib.load("application_estimator.joblib.pkl")
data_file=pandas.read_csv('Job_Pool_Data.csv')
name=numpy.array(data_file.iloc[:, 0])
data_size=numpy.array(data_file.iloc[:, 25])
X =numpy.array(data_file.iloc[:, 2:26])
a=[3.4	,4,	32,	4,	34.1,	870.4,	32,	2.133]
b=[0.993	,1.003	,384	,2	,28.8	,762	,2	,1.8]
c=[3.4	,4	,32	,4	,34.1	,870.4	,32	,2.133]
d=[0.993	,1.003	,384	,2	,28.8	,762	,2	,1.8]
e=[3.2,3.4,16,8,25.6,409.6,32,1.6]
f_1=[0.98	,1.033	,1152	,2	,192.2	,2257.9	,2	,6]
if os.path.exists("mapping.csv"):
    os.remove("mapping.csv")

f=open("mapping.csv","a")
f.write("Application_Name,Data_size,740_1_CPU,740_1_GPU,740_2_CPU,740_2_GPU,760_CPU,760_GPU, Device_Suitability\n")
f.close()

data=""
for u in range (0,len(X)):
	print u
	
	#my_ranking= model1.predict_proba(X[u].reshape(1,-1))
	#data+=str(name[u])+","+str(data_size[u])+","+str(my_ranking[0][0])+","+str(my_ranking[0][1])+","+str(my_ranking[0][2])+","+str(my_ranking[0][3])+","+str(my_ranking[0][4])+","+str(my_ranking[0][5])+","+str(model1.predict(X[u].reshape(1,-1))).replace('[','').replace(']','')+"\n"
	my_ranking=rankdata(( model1.predict_proba(X[u].reshape(1,-1))),method="dense")
	#data+=str(name[u])+","+str(data_size[u])+","+str(my_ranking[0])+","+str(my_ranking[1])+","+str(my_ranking[2])+","+str(my_ranking[3])+","+str(my_ranking[4])+","+str(my_ranking[5])+","+str(model1.predict(X[u].reshape(1,-1))).replace('[','').replace(']','')+"\n"
	data+=str(name[u])+","+str(data_size[u])+","+str((model2.predict(numpy.concatenate((numpy.array((X[u],numpy.array(a))))).reshape(1,-1)))).replace('[','').replace(']','').replace('-','')+","+str(model2.predict(numpy.concatenate((numpy.array((X[u],numpy.array(b))))).reshape(1,-1))).replace('[','').replace(']','').replace('-','')+","+str(model2.predict(numpy.concatenate((numpy.array((X[u],numpy.array(c))))).reshape(1,-1))).replace('[','').replace(']','').replace('-','')+","+str(model2.predict(numpy.concatenate((numpy.array((X[u],numpy.array(d))))).reshape(1,-1))).replace('[','').replace(']','').replace('-','')+","+str(model2.predict(numpy.concatenate((numpy.array((X[u],numpy.array(e))))).reshape(1,-1))).replace('[','').replace(']','').replace('-','')+","+str(model2.predict(numpy.concatenate((numpy.array((X[u],numpy.array(f_1))))).reshape(1,-1))).replace('[','').replace(']','').replace('-','')+","+str(model1.predict(X[u].reshape(1,-1))).replace('[','').replace(']','').replace('-','')+"\n"
	f=open("mapping.csv","a")
	f.write(data)
	f.close()
	data=""


"""
#print model1.classes_
#print model1.predict(X[0].reshape(1,-1))

print len(r2)


print "Application_Estimator Class 1:",model2.predict(numpy.concatenate((numpy.array((X[0],numpy.array(a))))).reshape(1,-1))
print "Application_Estimator Class 2:",model2.predict(numpy.concatenate((numpy.array((X[0],numpy.array(b))))).reshape(1,-1))

print "Application_Estimator Class 3:",model2.predict(numpy.concatenate((numpy.array((X[0],numpy.array(c))))).reshape(1,-1))
print "Application_Estimator Class 4:",model2.predict(numpy.concatenate((numpy.array((X[0],numpy.array(d))))).reshape(1,-1))
print "Application_Estimator Class 5:",model2.predict(numpy.concatenate((numpy.array((X[0],numpy.array(e))))).reshape(1,-1))
print "Application_Estimator Class 6:",model2.predict(numpy.concatenate((numpy.array((X[0],numpy.array(f))))).reshape(1,-1))

1	740_1_CPU
2	740_1_GPU
3	740_2_CPU
4	740_2_GPU
5	760_CPU
6	760_GPU
ty=X_test[0].reshape(1, -1)
print ty
print model.classes_
print model.predict(ty)
print model.predict_proba(ty)


ty=X[0].reshape(1, -1)



predicted=[]
print len(predicted)
print len(predicted)
for d in range(0,len(ty)):
	predicted=predicted.append(ty[d])
for d in range(0,len(a)):
	predicted=predicted.append(a[d])
print len(predicted)

model2 = joblib.load("application_estimator.joblib.pkl")
print model2.predict(ty)


#mylist=[0.9,0.8,1,0.7]

#print numpy.argsort(mylist)
"""