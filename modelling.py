# @author presario
# @date April 4, 2017
# @y070049@gmail.com
import pandas as pd
import math
import numpy as np
#from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler,  RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
#from sklearn.decomposition import TruncatedSVD
#from sklearn.random_projection import sparse_random_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Utility function to report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def readData():
    dropFields = ['loan_status']
    all = pd.read_csv('loan_2010_12_with_dummy.csv')
    X = (all.drop(dropFields, axis=1))
    y = all['loan_status']

    # split training and test 80% : 20%
    # training using 5-fold cross validation i.e. validation 20% of the total dataset
    sss = StratifiedShuffleSplit(test_size=0.2)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]


    scaler = StandardScaler()
    scaler.fit(X_train)
    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print(X_train)

    return X_train, y_train, X_test, y_test

def logisticRegression():

    X_train, y_train, X_test, y_test = readData()

    # logistic regression
    lr = LogisticRegression(class_weight='balanced', random_state=10, C=1,
                            penalty='l2', dual=False,
                            solver='liblinear', n_jobs=-1
                            )
    lr.fit(X_train, y_train)

    scores = cross_val_score(lr, X_train, y_train, cv=5)
    print(lr.coef_)
    print(scores)
    lr_preds = lr.predict(X_test)
    score = accuracy_score(y_test, lr_preds)
    print(score)
    c1 = confusion_matrix(y_test, lr_preds, labels=[1, 0])
    print(c1)

    lr_fpr, lr_tpr, lr_threholds = roc_curve(y_test, lr_preds)
    lr_auc = auc(lr_fpr, lr_tpr)

    return lr_fpr, lr_tpr, lr_auc

def gridSearchCV(rf, X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=150,
        oob_score=True,
        criterion='gini',
        max_features=15,
        # max_depth=10,
        n_jobs=4,
        random_state=10,
        min_samples_leaf=1,
        class_weight='balanced',
        warm_start=False
    )

    # use a full grid over all parameters
    param_grid = {
        # 'max_features': [5, 10, 15, 20, 25],
        "max_depth": [5, 10, 20, 30, None],
        # "max_features": [5, 8, 12, 20]
    }
    # run grid search
    grid_search = GridSearchCV(rf, cv=5, param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    report(grid_search.cv_results_)

def randomForest():
    X_train, y_train, X_test, y_test = readData()

    # random forest
    # use dataset without creating dummy variables
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, criterion='gini', max_features=8,
                                max_depth=10,
                                n_jobs=-1, random_state=10, min_samples_leaf=1,
                                class_weight='balanced', warm_start=False
                                )


    # change above rf parameters and run grid search
    #gridSearchCV(rf, X_train, y_train)

    rf.fit(X_train, y_train)
    scores = cross_val_score(rf, X_train, y_train, cv=5)
    print(scores)
    preds = rf.predict(X_test)

    score = accuracy_score(y_test, preds)
    print(score)

    c2 = confusion_matrix(y_test, preds, labels=[1, 0])
    print(c2)

    rf_fpr, rf_tpr, rf_threholds = roc_curve(y_test, preds)
    rf_auc = auc(rf_fpr, rf_tpr)

    return rf_fpr, rf_tpr, rf_auc

def mlp():
    X_train, y_train, X_test, y_test = readData()
    mlp = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10,), learning_rate_init=0.01, random_state=1,
                        solver='sgd', tol=0.000001, verbose=False, warm_start=True)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    print(mlp.score(X_test, y_test))
    print(accuracy_score(y_test, predictions))

    c2 = confusion_matrix(y_test, predictions, labels=[1, 0])
    print(c2)

    mlp_fpr, mlp_tpr, mlp_threholds = roc_curve(y_test, predictions)
    mlp_auc = auc(mlp_fpr, mlp_tpr)

    return mlp_fpr, mlp_tpr, mlp_auc

def svm():
    X_train, y_train, X_test, y_test = readData()
    svc = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False
              )
    svc.fit(X_train, y_train)
    predictions = svc.predict(X_test)
    print(svc.score(X_test, y_test))
    print(accuracy_score(y_test, predictions))

    c2 = confusion_matrix(y_test, predictions, labels=[1, 0])
    print(c2)

    svm_fpr, svm_tpr, svm_threholds = roc_curve(y_test, predictions)
    svm_auc = auc(svm_fpr, svm_tpr)

    return svm_fpr, svm_tpr, svm_auc

def compareWithROC():
    rf_fpr, rf_tpr, rf_auc = randomForest()
    lr_fpr, lr_tpr, lr_auc = logisticRegression()
    mlp_fpr, mlp_tpr, mlp_auc = mlp()
    svm_fpr, svm_tpr, svm_auc = svm()

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(rf_fpr, rf_tpr, label='Random Forest AUC = %0.2f'% rf_auc)
    plt.plot(lr_fpr, lr_tpr, label='Logistic Regression AUC = %0.2f'% lr_auc)
    plt.plot(mlp_fpr, mlp_tpr, label='Multilayer Perceptron AUC = %0.2f' % mlp_auc)
    plt.plot(svm_fpr, svm_tpr, label='C-Support Vector AUC = %0.2f' % svm_auc)

    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison')
    plt.legend(loc='best')
    plt.show()


def main():
    #logisticRegression()
    #randomForest()
    compareWithROC()

if __name__ == '__main__':
    main()