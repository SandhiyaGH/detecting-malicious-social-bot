
import glob
import string
import ast

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier, export_graphviz 
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


#metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold




train = pd.DataFrame()

with open('Preprocessed/Train.csv', encoding="utf8", errors='ignore') as f:
    df = pd.read_csv(f)  
    train = train.append(df,ignore_index=True) 
    
test = pd.DataFrame()
with open('Preprocessed/Test.csv', encoding="utf8", errors='ignore') as f:
    df = pd.read_csv(f)
    test = test.append(df,ignore_index=True)  
validSet = pd.DataFrame()
with open('Preprocessed/Validation.csv', encoding="utf8", errors='ignore') as f:

    df = pd.read_csv(f)
    validSet = validSet.append(df,ignore_index=True)
    
print(train.shape, validSet.shape, test.shape)


print(train.info())

xset = train
X_all = xset.drop(['id','bot','name'], axis=1)
y_all = xset['bot']

# Create the RFE object and compute a cross-validated score.
model = ExtraTreesClassifier()
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(10),scoring='accuracy')
rfecv.fit(X_all, y_all)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


from sklearn.ensemble import ExtraTreesClassifier
model2 = ExtraTreesClassifier()
model2.fit(X_all, y_all)

#Add features and their importances to a dictionary
feature_imp_dict = dict(zip(X_all.columns.values, model2.feature_importances_))

for x in sorted(zip(model2.feature_importances_,list(X_all)))[::-1]:
    print( x)
