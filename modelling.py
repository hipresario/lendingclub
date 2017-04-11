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

def logisticRegression():
    dropFields = ['loan_status', 'int_rate', 'annual_inc', 'revol_bal', 'installment']
    all = pd.read_csv('loan_2010_12_with_dummy.csv')
    X = (all.drop(dropFields, axis=1))
    y = all['loan_status']

    # split training and test 80% : 20%
    # training using 4-fold cross validation i.e. validation 20% of the total dataset
    sss = StratifiedShuffleSplit(test_size=0.2)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    scaler.fit(X_train)
    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # logistic regression
    lr = LogisticRegression(class_weight='balanced', random_state=10, C=1,
                            penalty='l2', dual=False,
                            solver='liblinear', n_jobs=-1
                            )
    lr.fit(X_train, y_train)
    scores = cross_val_score(lr, X_train, y_train, cv=5)
    print(lr.coef_)
    print(scores)
    test_pred = lr.predict(X_test)
    score = accuracy_score(y_test, test_pred)
    print(score)
    c1 = confusion_matrix(y_test, test_pred, labels=[1, 0])
    print(c1)

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
    dropFields = ['loan_status', 'int_rate', 'annual_inc', 'revol_bal', 'installment']
    all = pd.read_csv('loan_2010_12_with_dummy.csv')
    X = (all.drop(dropFields, axis=1))
    y = all['loan_status']

    # split training and test 80% : 20%
    # training using 4-fold cross validation i.e. validation 20% of the total dataset
    sss = StratifiedShuffleSplit(test_size=0.2)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # # Now apply the transformations to the data:
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # random forest
    # use dataset without creating dummy variables
    rf = RandomForestClassifier(n_estimators=150, oob_score=True, criterion='gini', max_features=10,
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

def main():
    #logisticRegression()
    randomForest()

if __name__ == '__main__':
    main()