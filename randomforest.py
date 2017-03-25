# @author presario
# @date Mar 12, 2017
# @y070049@gmail.com
import pandas as pd
#from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler,  RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
#from sklearn.decomposition import TruncatedSVD
#from sklearn.random_projection import sparse_random_matrix
from sklearn.metrics import accuracy_score

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
    all = pd.read_csv('train.csv')
    X = (all.drop(['Prediction'], axis=1))
    #X['NextId'] = X[X['NextId'] == -1]['NextId'] =X['Id'] + 1
    #X['Id'] = X['Id'] / 52152
    #X['NextId'] = X['NextId']/52152
    #X['Position'] = X['Position'] / 14
    y = all['Prediction']

    #X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1)

    sss = StratifiedShuffleSplit(test_size=0.1)

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

    #max_features = [65]  # 10, 30, 50, 70, 100 # best 0.93 estimators= 160, max_features= 60, max_depth = 40
    #for leaf_size in max_features:
    rf = RandomForestClassifier(n_estimators=300, oob_score=True, criterion='gini', max_features=68,
                                max_depth = 40,
                                n_jobs=-1, random_state=100, min_samples_leaf=1,
                                class_weight='balanced', warm_start=False
                                )

    rf.fit(X_train, y_train)
    print(rf.score(X_train, y_train))
    test_pred = rf.predict(X_test)
    score = accuracy_score(y_test, test_pred)
    print(score)

    final_all = pd.read_csv('test.csv')
    x_final = (final_all.drop(['Prediction'], axis=1))
    y_pre = rf.predict(x_final)

    pd.DataFrame().assign(
        Id=x_final['Id'],
        Prediction=y_pre
    ).to_csv("test_final.csv", index=False)

    output()

if __name__ == '__main__':
    main()