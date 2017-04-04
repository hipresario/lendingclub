# @author presario
# @date April 4, 2017
# @y070049@gmail.com
import pandas as pd
import math
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
from sklearn.model_selection import StratifiedKFold

def output():
    test = pd.read_csv('test_final.csv')
    test = pd.DataFrame().assign(
        Id=test['Id'],
        Prediction=test['Prediction']
    )
    train = pd.read_csv('train.csv')
    result = pd.DataFrame().assign(
        Id=train['Id'],
        Prediction=train['Prediction']
    )
    submit = pd.concat([test, result])
    ans = submit.sort(['Id'], ascending=[1])
    # print(ans)
    pd.DataFrame().assign(
        Id=ans['Id'],
        Prediction=ans['Prediction']
    ).to_csv('final_result.csv', index=False)

def main():
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

    #logistic regression
    lr = LogisticRegression( class_weight='balanced', random_state = 10, C=0.5,
                             penalty='l2',dual = False,
                             solver = 'liblinear',n_jobs = -1
                             )
    #
    lr.fit(X_train, y_train)
    scores = cross_val_score(lr, X_train, y_train, cv=4)
    #print(lr.score(X_train, y_train))
    print(lr.coef_)
    print(scores)
    test_pred = lr.predict(X_test)
    score = accuracy_score(y_test, test_pred)
    print(score)
    c1 = confusion_matrix(y_test, test_pred, labels = ['GOOD', 'BAD'])
    print(c1)
    return
    # X_test['expected'] = y_test
    # X_test['predict'] = test_pred
    # X_test.to_csv('result.csv', index=False)
    #

    #random forest
    rf = RandomForestClassifier(n_estimators=60, oob_score=True, criterion='gini', max_features=4,
                                max_depth = 8,
                                n_jobs=-1, random_state=10, min_samples_leaf=1,
                                class_weight='balanced', warm_start=False
                                )
    rf.fit(X_train,y_train)
    print(rf.score(X_train, y_train))
    preds = rf.predict(X_test)
    score = accuracy_score(y_test, preds)
    c2 =confusion_matrix(y_test, preds, labels=['GOOD', 'BAD'])
    print(c2)

if __name__ == '__main__':
    main()