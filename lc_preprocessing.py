import pandas as pd
import math

def cleanData(src):
    """
    remove loans with current loan status
    :param src:
    :return:
    """
    rawdata = pd.read_csv(src, encoding='latin-1')
    cleandata = rawdata[rawdata['loan_status'] != 'Current']
    cleandata.to_csv('loan_clean.csv', index=False, encoding='latin-1' )

#selected attributes for experiment
fields = ['loan_amnt', 'installment','sub_grade',
          'emp_length','home_ownership','annual_inc',
           'loan_status',
          'purpose','dti','delinq_2yrs',
          'earliest_cr_line','inq_last_6mths','open_acc',
          'revol_bal','revol_util']

#convert grades to numericals
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

#convert years to numericals
YEARS = [' ', 'n/a', '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years',
         '8 years', '9 years', '10 years', '10+ years']
YEARS_VALUE = [0, 0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]

#prediction class conversion
LOAN_STATUS = ['Default', 'Charged Off', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)', 'Fully Paid']
LOAN_STATUS_VALUE = [0, 0, 0, 0, 0, 1]

def fixYear(x):
    """
    fix employment length
    :param x:
    :return:
    """
    if not x or x == '':
        return 'n/a'
    return x

def calculateCreditYear(row):
    """
    convert earliest credit years
    :param row:
    :return:
    """
    year = str(row['earliest_cr_line'])
    if not year or year.strip() == '':
        return 'n/a'
    years = year.split('-')
    num = float(years[1][0])
    if num > 2:
        years = '19' + years[1]
    else:
        years = '20' + years[1]
    dif = 2012 - float(years)

    return dif

def p2f(x):
    """
    percentage strings to numbers in float
    :param x:
    :return:
    """
    if not x or x == 'n/a' or x == '':
        return 0
    x = x.strip('%')
    return float(x)/100

def selectData(src):
    """
    filter selected data only
    :param src:
    :return:
    """
    df = pd.read_csv(src, encoding='latin-1',  skipinitialspace=True, usecols=fields,
                     na_values = {'n/a','na', ''},
                    converters={'int_rate':p2f,
                                'revol_util': p2f,
                                'emp_length': fixYear
                                })

    df['credit_years'] = df.apply(calculateCreditYear, axis=1)

    df['emp_length'].replace(
        to_replace=YEARS,
        value=YEARS_VALUE,
        inplace=True
    )

    df['loan_status'].replace(
        to_replace=LOAN_STATUS,
        value=LOAN_STATUS_VALUE,
        inplace=True
    )

    df['sub_grade'].replace(
        to_replace=SUB_GRADE,
        value=SUB_GRADE_VALUE,
        inplace=True
    )



    df.to_csv('loan_2010_12_clean.csv', index=False)

def createDummyVar():
    """
    create dummmy variables
    :return:
    """
    df = pd.read_csv('loan_2010_12_clean.csv', encoding='latin-1')
    df = (df.drop(['earliest_cr_line'], axis=1))
    df['installment_to_income'] = (12 * df['installment'])/ df['annual_inc']
    df['revol_to_income'] = (df['revol_bal']) / df['annual_inc']
    df = pd.get_dummies(df, prefix='home_', columns=['home_ownership'])
    df = pd.get_dummies(df, prefix='purpose_', columns=['purpose'])

    df.to_csv('loan_2010_12_with_dummy.csv', index=False)

def formatData():
    all = pd.read_csv('loan_2010_12_with_dummy.csv', encoding='latin-1', dtype='float')
    all.to_csv('loan_2010_12_with_dummy_clean.csv', index=False)

def main():
    #clean data
    src = 'loan.csv'
    cleanData(src)
    src = 'loan_clean.csv'
    selectData(src)
    createDummyVar()
    formatData()


if __name__ == '__main__':
    main()