import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
import  math

def cleanData(num):
    src = 'LoanStats_2016Q' + num+ '.csv'
    rawdata = pd.read_csv(src, encoding='latin-1')
    cleandata = rawdata[rawdata['loan_status'] != 'Current']
    cleandata.to_csv('2016Q'+ num + '.csv', index=False, encoding="latin-1" )

fields = ['loan_amnt', 'int_rate', 'sub_grade', 'term',
          'emp_length', 'home_ownership', 'annual_inc',
          'verification_status',	'loan_status',
          'purpose', 'dti',	'delinq_2yrs',
          'inq_last_6mths', 'open_acc',
          'pub_rec',	'revol_bal',	'revol_util',
          'initial_list_status', 'application_type',
          'avg_cur_bal',	'pub_rec_bankruptcies']

HOME = ['MORTGAGE', 'RENT',  'OWN']
HOME_VALUE = [1, 2, 3]

PURPOSE = ['car', 'credit_card', 'debt_consolidation', 'home_improvement', 'house',
            'major_purchase', 'medical', 'moving', 'renewable_energy', 'small_business',
           'vacation', 'other'
            ]
PURPOSE_VALUE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

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
APPLICATION_TYPE = [ 'INDIVIDUAL', 'JOINT', 'DIRECT_PAY']
APPLICATION_TYPE_VALUE = [0, 1, 2]

YEARS = ['', 'n/a', '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years',
         '8 years', '9 years', '10 years', '10+ years']
YEARS_VALUE = [0, 0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]

LOAN_STATUS = ['Default', 'Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)', 'Fully Paid']
LOAN_STATUS_VALUE = ['BAD', 'BAD', 'BAD', 'BAD', 'BAD', 'GOOD']


def p2f(x):
    if x == 'n/a':
        return 0
    x = x.strip('%')
    if not x:
        return 0
    return float(x)/100


def selectData(num):
    src = '2016Q' + num + '.csv'
    df = pd.read_csv(src, encoding='latin-1',  skipinitialspace=True, usecols=fields,
                     na_values = {'n/a','na', ''},
                    converters={'int_rate':p2f, 'revol_util': p2f})

    df['term'].replace(
        to_replace=[36, 60],
        value=[1,2],
        inplace=True
    )

    df['initial_list_status'].replace(
        to_replace=['w', 'f'],
        value=[1, 2],
        inplace=True
    )
    df['emp_length'].replace(
        to_replace=YEARS,
        value=YEARS_VALUE,
        inplace=True
    )
    df['verification_status'].replace(
        to_replace=['Source Verified', 'Verified', 'Not Verified'],
        value=[1, 1, 2],
        inplace=True
    )
    df['loan_status'].replace(
        to_replace=LOAN_STATUS,
        value=LOAN_STATUS_VALUE,
        inplace=True
    )

    df['home_ownership'].replace(
        to_replace=HOME,
        value=HOME_VALUE,
        inplace=True
    )
    df['purpose'].replace(
        to_replace=PURPOSE,
        value=PURPOSE_VALUE,
        inplace=True
     )
    df['sub_grade'].replace(
        to_replace=SUB_GRADE,
        value=SUB_GRADE_VALUE,
        inplace=True
    )
    df['application_type'].replace(
        to_replace=APPLICATION_TYPE,
        value=APPLICATION_TYPE_VALUE,
        inplace=True
     )
    df.to_csv('2016Q' + num + '_ALL.csv', index=False)

def combineData():
    srcs = ['1', '2', '3', '4']
    frames = []
    for src in srcs:
        src = '2016Q' + src + '_All.csv'
        df = pd.read_csv(src, encoding='latin-1')
        frames.append(df)

    results = pd.concat(frames)
    results.to_csv('2016_LC.csv', index=False)


def main():
    #clean data
    srcs = ['1', '2', '3', '4']
    # for src in srcs:
    #     # cleanData(src)
    #     selectData(src)
    # #statistics()
    #combineData()



    # df = pd.read_csv('2016Q1_All.csv', encoding='latin-1')
    # fixEmployment(df)

def statistics():
    q1 = pd.read_csv('2016Q1_All.csv', encoding='latin-1')
    print(q1.describe())



if __name__ == '__main__':
    main()