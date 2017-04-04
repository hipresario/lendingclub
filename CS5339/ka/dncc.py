import tensorflow as tf
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)


# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
	'processed_train_data.csv',
    	target_dtype=np.int,
    	features_dtype=np.float32,
	target_column=1)

test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
	'processed_test_data.csv',
    	target_dtype=np.int,
    	features_dtype=np.float32,
	target_column=1)

predict_data=np.genfromtxt('processed_predict_data.csv', dtype='float', delimiter=',')


# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=111)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[111, 140, 26],
                                            n_classes=26, 
                                            model_dir="/tmp/cs5339_model")

print('=============================')

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=5000)

accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('=========================Accuracy: {0:f}'.format(accuracy_score))


result = classifier.predict(predict_data)
classification=np.asarray(list(result))

classification = classification.reshape(10473,1)

test_ids=np.genfromtxt('test.csv', dtype='string', delimiter=',')
test_ids=test_ids[:,[0]]
test_ids=np.delete(test_ids,0,0)
rr=np.concatenate((test_ids, classification), axis=1)
for row in rr:
	asciic=int(row[1]) + 97
	if asciic >=0 and asciic <=256:
		row[1]= chr(asciic)
	else:
		print (asciic)

result_data=np.genfromtxt('train.csv', dtype='string', delimiter=',')
result_data=result_data[:,[0,1]]
result_data=np.concatenate((result_data, rr), axis=0)   
np.savetxt('ChenShaozhuang-A0134531R.csv', result_data, fmt='%s', delimiter=',')  
