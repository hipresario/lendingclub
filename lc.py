import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
import math

def cleanData(src):
    rawdata = pd.read_csv(src, encoding='latin-1')
    cleandata = rawdata[rawdata['loan_status'] != 'Current']
    cleandata.to_csv('loan_clean.csv', index=False, encoding="latin-1" )

fields = ['loan_amnt','int_rate','installment','sub_grade',
          'emp_length','home_ownership','annual_inc',
          'verification_status','loan_status',
          'purpose','title','dti','delinq_2yrs',
          'earliest_cr_line','inq_last_6mths','open_acc',
          'revol_bal','revol_util','initial_list_status']

HOME = ['MORTGAGE', 'RENT',  'OWN']
HOME_VALUE = [1, 2, 3]

PURPOSE = ['car', 'credit_card', 'debt_consolidation', 'home_improvement', 'house',
            'major_purchase', 'medical', 'moving', 'renewable_energy', 'small_business',
           'vacation', 'wedding', 'other'
            ]
PURPOSE_VALUE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

SUB_GRADE = ['A1', 'A2', 'A3', 'A4', 'A5',
             'B1', 'B2', 'B3', 'B4', 'B5',
             'C1', 'C2', 'C3', 'C4', 'C5',
             'D1', 'D2', 'D3', 'D4', 'D5',
             'E1', 'E2', 'E3', 'E4', 'E5',
             'F1', 'F2', 'F3', 'F4', 'F5',
             'G1', 'G2', 'G3', 'G4', 'G5'
             ]
SUB_GRADE_VALUE = [1, 2, 3, 4, 5,
                   6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25,
                   26, 27, 28, 29, 30,
                   31, 32, 33, 34, 35
                   ]

YEARS = [' ', 'n/a', '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years',
         '8 years', '9 years', '10 years', '10+ years']
YEARS_VALUE = [0, 0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]

LOAN_STATUS = ['Default', 'Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)', 'Fully Paid']
LOAN_STATUS_VALUE = ['BAD', 'BAD', 'BAD', 'BAD', 'BAD', 'GOOD']

def fixYear(x):
    if not x or x == '':
        return 'n/a'
    return  x;

def p2f(x):
    if not x or x == 'n/a' or x == '':
        return 0
    x = x.strip('%')
    return float(x)/100

def titleLength(x):
    return len(str(x))

def selectData(src):
    df = pd.read_csv(src, encoding='latin-1',  skipinitialspace=True, usecols=fields,
                     na_values = {'n/a','na', ''},
                    converters={'int_rate':p2f,
                                'revol_util': p2f,
                                'emp_length': fixYear,
                                'title': titleLength})

    df['initial_list_status'].replace(
        to_replace=['w', 'f'],
        value=[1, 0],
        inplace=True
    )
    df['emp_length'].replace(
        to_replace=YEARS,
        value=YEARS_VALUE,
        inplace=True
    )
    df['verification_status'].replace(
        to_replace=['Source Verified', 'Verified', 'Not Verified'],
        value=[0, 0, 1],
        inplace=True
    )
    df['loan_status'].replace(
        to_replace=LOAN_STATUS,
        value=LOAN_STATUS_VALUE,
        inplace=True
    )

    # df['home_ownership'].replace(
    #     to_replace=HOME,
    #     value=HOME_VALUE,
    #     inplace=True
    # )
    # df['purpose'].replace(
    #     to_replace=PURPOSE,
    #     value=PURPOSE_VALUE,
    #     inplace=True
    #  )
    df['sub_grade'].replace(
        to_replace=SUB_GRADE,
        value=SUB_GRADE_VALUE,
        inplace=True
    )

    df.to_csv('loan_2010_12.csv', index=False)

def combineData():
    srcs = ['1', '2', '3', '4']
    frames = []
    for src in srcs:
        src = '2016Q' + src + '_All.csv'
        df = pd.read_csv(src, encoding='latin-1')
        frames.append(df)

    results = pd.concat(frames)
    results.to_csv('2016_LC.csv', index=False)

def splitData():
    """Training, Validation, Test 6:2:2"""
    all = pd.read_csv('2016_LC.csv', encoding='latin-1')
    X = (all.drop(['loan_status'], axis=1))
    y = all['loan_status']

    sss = StratifiedShuffleSplit(test_size=0.2)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

    X_train = X_train.copy()
    X_train['loan_status'] = y_train

    X_test = X_test.copy()
    X_test['loan_status'] = y_test

    X_train.to_csv('2016_train.csv', index=False)
    X_test.to_csv('2016_test.csv', index=False)


def main():
    #clean data
    #src = 'loan.csv'
    #cleanData(src)
    src = 'loan_clean.csv'
    selectData(src)
    # #statistics()
    #combineData()
    #splitData()


    # df = pd.read_csv('2016Q1_All.csv', encoding='latin-1')
    # fixEmployment(df)

def statistics():
    q1 = pd.read_csv('2016Q1_All.csv', encoding='latin-1')
    print(q1.describe())



if __name__ == '__main__':
    main()