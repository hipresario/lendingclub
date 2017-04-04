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

def extract_data(file_name):
	my_data=np.genfromtxt(file_name, dtype='object', delimiter=',')
	
	#remove header
	my_data=np.delete(my_data, 0, axis=0)

	rows=my_data.shape[0]
	for x in range(0, rows):
		row=my_data[x]
		row[2]=find_next_char(x, my_data)
		row[3]=0

		for byte in range(16):
			row[4+byte]=byte_to_number(my_data, x, 4+byte*8) 

	result_data=np.delete(my_data, [0,1,2,3], axis=1)
	result_data=result_data[:, 0:16]

	return result_data
	
#extract train data.
train_data=extract_data('train.csv')

#extract test data
test_data=extract_data('test.csv')

#extract train label.
my_data=np.genfromtxt('train.csv', dtype='object', delimiter=',')
my_data=np.delete(my_data, 0, axis=0)
train_labels=my_data[:,[1]]
for row in train_labels:
	row[0]= ord(row[0])-97
	if row[0]>25 or row[0]<0:
		print 'Error data:'+row[0]


np.savetxt('processed_train_data.csv', train_data, fmt='%s', delimiter=',')
np.savetxt('processed_train_labels.csv', train_labels, fmt='%s', delimiter=',')

np.savetxt('processed_test_data.csv', test_data, fmt='%s', delimiter=',')
