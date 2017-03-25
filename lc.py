import pandas as pd
import os


def cleanData(num):
    src = 'LoanStats_2016Q' + num+ '.csv'
    rawdata = pd.read_csv(src, encoding='latin-1')
    cleandata = rawdata[rawdata['loan_status'] != 'Current']
    cleandata.to_csv('2016Q'+ num + '.csv', index=False )

def main():
    #clean data
    srcs = ['1', '2', '3', '4']
    for src in srcs:
        cleanData(src)




if __name__ == '__main__':
    main()