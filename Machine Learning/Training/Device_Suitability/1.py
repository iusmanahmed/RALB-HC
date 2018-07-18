
import numpy as np  
import scipy.io as sp  
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation 
from sklearn import tree
from pandas_confusion import ConfusionMatrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Gradient Boosting Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png', dpi=200, format='png', bbox_inches='tight')


data_file=pd.read_csv('device_suitability.csv')
trainx= data_file.iloc[:, 2:26]
#print trainx
trainy= data_file.iloc[:, 26]
#print trainy
'''
print trainx[0:1]

# Correction Matrix Plot
import matplotlib.pyplot as plt
import pandas
import numpy

#names=['Cache Misses', 'Task Clock',   'CPUs Utilized' ,  ' Context Switches' ,'CPU Migrations'  , 'Page Faults',  'Cycles', ' instructions'   , 'Branches'   , 'branchMisses'   , 'secondsTimeElapsed',"Out"]
names=["[0]Data Size",  "[1]- Total no of Return statement" ,"[2]- Total no of Control Statement"," [3]- Total no of Allocation instruction","  [4]- Total no of Load Instructions","   [5]- Total no of Store Instructions","[6]- Total no of Multiplication (Float Datatype) Operation"," [7]- Total no of Addition(Integer Datatype) Instruction","  [8]- Total no of Multiplication(Integer Datatype) Instruction","    [9]- Total no of Division(Float Datatype) instruction","    [10]-Total no of Division(Integer Datatype) instruction","  [11]-Total no of Condition Check instruction","[12]-Total no of Addition(Float Datatype) instruction:","    [13]-Total no of Addition(Integer Datatype) instruction","  [14]-Total no of Subtraction(Float Datatype) instruction"," [15]-Total no of Subtraction(Integer Datatype) instruction","   [16]-Total no of Function Call instruction","   [17]-Total no of Functions","   [18]-Total no of Blocks","  [19]-Total no of Instructions","    [20]-Total no of Float Operation"," [21]-Total no of Integer Operation","   [22]-Total no of Loop Operation","  [23]-Total no of Loop"]

correlations = trainx.corr()
print correlations
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
for (i, j), z in np.ndenumerate(correlations):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
fig.colorbar(cax)
#ticks = numpy.arange(0,9,1)
#print ticks
ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(ticks,rotation='vertical')
ax.set_yticklabels(ticks)
plt.show()


from pandas.tools.plotting import scatter_matrix
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('Full_Data_Raw',names=names)
data = data.astype(np.float)
#scatter_matrix(data)
axs = scatter_matrix( data, alpha=0.2, diagonal='kde')
n = len(data.columns)
for x in range(n):
    for y in range(n):
        # to get the axis of subplots
        ax = axs[x, y]
        # to make x axis name vertical  
        ax.xaxis.label.set_rotation(90)
        # to make y axis name horizontal 
        ax.yaxis.label.set_rotation(0)
        # to make sure y axis names are outside the plot area
        ax.yaxis.labelpad = 50
plt.show()
'''
#print trainy[0:10]
print len(trainx)
ros = SMOTE(k_neighbors=4)
trainx,trainy= ros.fit_sample(trainx, trainy)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print len(trainx)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(trainx, trainy, train_size=0.70 ,test_size=0.30, random_state=42)
#model= tree.DecisionTreeClassifier(criterion='gini')
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler
from tpot.builtins import StackingEstimator, ZeroCount
model = make_pipeline(
    StackingEstimator(estimator=GaussianNB()),
    PCA(iterated_power=7, svd_solver="randomized"),
    XGBClassifier(learning_rate=0.5, max_depth=8, min_child_weight=1, n_estimators=100, nthread=1, subsample=0.5)
)

from sklearn.externals import joblib
model.fit(trainx, trainy)
print("actual ",model.feature_importances_)
joblib.dump(model, "device_suitability.joblib.pkl", compress=9)
#model=GradientBoostingClassifier()


"""
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, trainx, trainy, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""
#model=svm.SVC(decision_function_shape='ovo')
#model=make_pipeline(StandardScaler(), PCA(n_components=2), GradientBoostingClassifier())
#model=GradientBoostingClassifier()
#model=RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.7, min_samples_leaf=5, min_samples_split=13, n_estimators=100)
#model=tree.DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=10, min_samples_split=7)
#model = XGBClassifier()
model.fit(X_train, Y_train.ravel())
#print "feature_importances_"
#print(model.feature_importances_)
result = model.predict(X_test)
ty=X_test[0].reshape(1, -1)
print ty
print model.classes_
print model.predict(ty)
print model.predict_proba(ty)
import scikitplot as skplt
import matplotlib.pyplot as plt

#y_true = # ground truth labels
y_probas = model.predict_proba(X_test)
skplt.metrics.plot_roc_curve(Y_test, y_probas,text_fontsize=16)
plt.savefig('ROC.png', dpi=200, format='png', bbox_inches='tight')

skplt.metrics.plot_precision_recall_curve(Y_test, y_probas,text_fontsize=16,title_fontsize=16)
plt.savefig('Precision_recall.png', dpi=200, format='png', bbox_inches='tight')









import matplotlib.pyplot as plt
import numpy as np

def show_values(pc, fmt="%.2f", **kw):
   
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
  
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):

    fig, ax = plt.subplots()    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      
    plt.xlim( (0, AUC.shape[1]) )
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    plt.colorbar(c)
    show_values(c)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    lines = classification_report.split('\n')
    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)



def main(sampleClassificationReport):

    plot_classification_report(sampleClassificationReport)
    plt.savefig('test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()



print(classification_report(Y_test, result))



main(classification_report(Y_test, result))

class_names = ["1","2","3","4","5"]
cnf_matrix = confusion_matrix(Y_test, result)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
#plt.show()
