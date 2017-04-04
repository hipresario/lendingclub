import numpy as np

def find_next_char(row, data):
	next_id=int(data[row][2])
	if next_id != -1:
		return ord(data[row+1][1]) - 97
	return 0

def byte_to_number(data, row, index):
	binary=''
	for i in range(index, index+8):
		binary+=str(data[row][i])
	num=int(binary,2) 
	return round(1.0*num/256, 2)

def normalize_col(data, column, digit):
	col_data=data[:,[column]].astype(np.float32)
	col_data=col_data/col_data.max(axis=0)
	#col_data=np.round(col_data, decimals=4)
	rows=col_data.shape[0]
	for x in range(0, rows):
		row=col_data[x]
		row[0]=round(row[0], digit)
		row2=data[x]
		row2[column]=row[0]
		#print(row[0])
	#data[:,[column]]=col_data

def normalize_pos(data, column):
	col_data=data[:,[column]].astype(np.float32)
	

def extract_data(file_name):
	my_data=np.genfromtxt(file_name, dtype='object', delimiter=',')
	
	#remove header
	my_data=np.delete(my_data, 0, axis=0)

	rows=my_data.shape[0]
	for x in range(0, rows):
		row=my_data[x]
		row[1]=ord(row[1])-97

	normalize_col(my_data, 0, 5)
	normalize_col(my_data, 2, 5)
	#normalize_col(my_data, 3, 2)

	#result_data=np.delete(my_data, [], axis=1)
	result_data=my_data
	#108+4=112
	result_data=result_data[:, 0:112]

	return result_data
	
#extract train data.
all_data=extract_data('train.csv')

rows=all_data.shape[0]
indices = np.random.permutation(rows)
train_idx, test_idx = indices[:rows-3000], indices[rows-3000:]
train_data, test_data = all_data[train_idx,:], all_data[test_idx,:]

#extract test data
predict_data=extract_data('test.csv')
predict_data=np.delete(predict_data, [1], axis=1)

np.savetxt('processed_train_data.csv', train_data, fmt='%s', delimiter=',')
np.savetxt('processed_test_data.csv', test_data, fmt='%s', delimiter=',')

np.savetxt('processed_predict_data.csv', predict_data, fmt='%s', delimiter=',')
