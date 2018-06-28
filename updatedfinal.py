import keras
import keras as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras import optimizers
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.layers import Dense, Dropout, Activation, Flatten
import subprocess
import matplotlib.pyplot
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Model
from sklearn import svm
from sklearn.metrics import accuracy_score
from keras.models import model_from_json
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.utils import shuffle

<<<<<<< HEAD
filePath = '/data/Mihir'
=======
<<<<<<< HEAD
filePath = '/data/Mihir'
=======
# filePath = '/data/Mihir'
filePath = '..'


>>>>>>> db59719d3b7d9d0a822b133c0cab79e49eda9524
>>>>>>> d8b5f6b1a7026caa18e5a01ae12d89eb99b6ef15
def model():
	global filePath
	tf.reset_default_graph()
	keras.backend.clear_session()
	model = Sequential()
	model.add(Conv2D(8, (3, 3), padding='same',
					 input_shape=(513, 800, 3)))
	model.add(Activation('relu'))
	model.add(Conv2D(8, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(16, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(16, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))

	model.add(Flatten())
	#model.add(Dense(10))
	model.add(Activation('relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(57))
	model.add(Activation('softmax'))

	model.summary()

	# initiate RMSprop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
				  optimizer=opt,
				  metrics=['accuracy'])
	model.load_weights(filePath + '/SpeakAI_data/my_model_weights.h5')

	return model
# rootdir = path to training data dir


def training_data(rootdir):
	spectograms = []
	spect_read = []
	spectograms_ids = []
	for subdir, dirs, files in os.walk(rootdir):
		print(subdir,dirs,files)
		for file in files:

			if file.endswith('png'): 
				try:
					x = plt.imread(subdir+'/'+file)
				except:
					continue 
				if str(x.shape) == '(513, 800, 3)': 
					spect_read.append(x)
					#print(subdir) 
					name = subdir.replace(rootdir, '')
					#print(name)
					#name = name.replace('/spects', "")
					spectograms_ids.append(name)
					spectograms.append(file)
	temp = []
	temp.append(spect_read)
	temp.append(spectograms_ids)
	return temp
	# print(y_train)
# rootdir1 = path to test data dir
def test_data(rootdir1):
	spectograms = []
	spect_read = []
	spectograms_ids = []
	for subdir, dirs, files in os.walk(rootdir1):
		for file in files:
			if file.endswith('png'): 
				try:
					x = plt.imread(subdir+'/'+file)
				except:
					continue
				if str(x.shape) == '(513, 800, 3)': 
					spect_read.append(x)
					name = subdir.replace(rootdir1, '')
					#name = name.replace('/spects', "")
					spectograms_ids.append(name)
					spectograms.append(file)
	temp = []
	temp.append(spect_read)
	temp.append(spectograms_ids)
	return temp
	# print(y_test)
def fit_data(y_train,y_test,x_train, model):
	encoder = LabelEncoder()
	y_temp_train = y_train
	encoder.fit(y_temp_train)
	encoded_Y = encoder.transform(y_temp_train)
	dummy_y = np_utils.to_categorical(encoded_Y)
	svm_x_train = []
	svm_y_train = []
<<<<<<< HEAD
	# y_temp2_train = y_test
	# encoder.fit(y_temp2_train)
	encoded_Y = encoder.transform(y_test)
=======
	y_temp2_train = y_test
	encoder.fit(y_temp2_train)
	encoded_Y = encoder.transform(y_temp2_train)
>>>>>>> d8b5f6b1a7026caa18e5a01ae12d89eb99b6ef15
	print(encoded_Y)
	dummy2_y = np_utils.to_categorical(encoded_Y)
	model2 = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
	for i in range(len(x_train)):
		x_1 = np.expand_dims(x_train[i], axis=0)
		flatten_2_features = model2.predict(x_1)
		svm_x_train.append(flatten_2_features)
		svm_y_train.append(dummy_y[i])

	temp = []
	temp.append(svm_x_train)
	temp.append(svm_y_train)
	temp.append(dummy2_y)
	return temp


def dense():
	model = Sequential()
	# x_train=x_train.reshape((1000,784))
	model.add(Dense(100,input_dim=399168,activation='elu' ,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
<<<<<<< HEAD
	# model.add(Dropout(0.2))
=======
<<<<<<< HEAD
	# model.add(Dropout(0.2))
=======
	model.add(Dropout(0.2))
>>>>>>> db59719d3b7d9d0a822b133c0cab79e49eda9524
>>>>>>> d8b5f6b1a7026caa18e5a01ae12d89eb99b6ef15
	model.add(Dense(50,activation="tanh"))
	model.add(Dense(4))
	model.add(Activation('softmax'))
	opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
	model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["categorical_accuracy"])
	#model.fit(zxtrain,test,epochs=50, verbose=0 )
	#model.summary()
	
	return model


def model_train(x_train,y_train,x_test, y_test, model):
	global filePath
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> d8b5f6b1a7026caa18e5a01ae12d89eb99b6ef15
	model.load_weights(filePath + "/SpeakAI_data/models/ff-5000-27.43.hdf5")
	checkpoint = ModelCheckpoint("/data/Mihir/SpeakAI_data/models/ff-{epoch:02d}-{val_loss:.2f}.hdf5",  mode='auto', period=10000, monitor='val_acc')
	callbacks_list = [checkpoint]
	model.fit(x_train,y_train,epochs=100001,verbose=1, callbacks=callbacks_list,validation_data=(x_test, y_test))
<<<<<<< HEAD
=======
=======
	model.load_weights(filePath + "/SpeakAI_data/models/ff-10000-81.23.hdf5")
	checkpoint = ModelCheckpoint("ff-{epoch:02d}-{val_loss:.2f}.hdf5",  mode='auto', period=2500, monitor='val_acc')
	callbacks_list = [checkpoint]
	model.fit(x_train,y_train,epochs=10001,verbose=1, callbacks=callbacks_list,validation_data=(x_test, y_test))
>>>>>>> db59719d3b7d9d0a822b133c0cab79e49eda9524
>>>>>>> d8b5f6b1a7026caa18e5a01ae12d89eb99b6ef15
	savedata(model)


def extractData(svm_x_train,svm_y_train):
	svm_x_train = np.array(svm_x_train)
	# clf = svm.SVC(kernel='rbf', class_weight='balanced',probability=True)
	dataset_size = len(svm_x_train)
	svm_x_train = np.array(svm_x_train).reshape(dataset_size,-1)
	svm_y_train = np.array(svm_y_train)
	svm_y_train = [np.where(r==1)[0][0] for r in svm_y_train]
	return svm_x_train,svm_y_train


def model_test(x_test, y_test, x_train, y_train):
	global filePath
	json_file = open(filePath + '/SpeakAI_data/ffModel22nd.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(filePath + "/SpeakAI_data/ff-100-4.86.hdf5")
	print("Loaded model from disk")
	opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)
 	loaded_model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["categorical_accuracy"])
	acc = loaded_model.evaluate(x_test, y_test)
	predictedVal = loaded_model.predict(x_test)
	print(np.argmax(y_test,1),np.argmax(predictedVal,1))
 	print("test accuracy",acc)
 	acc1 =loaded_model.evaluate(x_train,y_train)
 	print("train accuracy",acc1)


# 	print(accuracy_score(svm_y_test, predicted))

def savedata(model): 
	global filePath
	model_json = model.to_json()
	with open(filePath + "/SpeakAI_data/ffModel22nd.json", "w") as json_file:
    		json_file.write(model_json)
    	model.save_weights(filePath + "/SpeakAI_data/ffModel22nd.h5")
    	print("Save Model To Disk")


def preload():
	global svm_x_train,svm_y_train,svm_x_test,svm_y_test,filePath
	mod = model()
	rootdir = filePath + '/SpeakAI_data/data/train/'
	print(rootdir)
	traindata = training_data(rootdir)
	print(traindata)
	x_train = traindata[0]
	y_train = traindata[1]
	rootdir1 = filePath + '/SpeakAI_data/data/test/'
	testdata = test_data(rootdir1)
	x_test = testdata[0]
	y_test = testdata[1]
	svm_data = fit_data(y_train,y_test,x_train,mod)
	svm_x_train = svm_data[0]
	svm_y_train = svm_data[1]
	dum = svm_data[2]
	svm_x_train,svm_y_train = extractData(svm_x_train,svm_y_train)
	# clf = pickle.load(open('clf.p','rb'))
	svm_y_test = []
	svm_x_test = []
	model2 = Model(inputs=mod.input, outputs=mod.get_layer('flatten_1').output)


	for i in range(len(x_test)):
		x_1 = np.expand_dims(x_test[i], axis=0)
		#x_1 = preprocess_input(x_1)
		flatten_2_features = model2.predict(x_1)
		svm_x_test.append(flatten_2_features)
		svm_y_test.append(dum[i])
	svm_x_test = np.array(svm_x_test)
	dataset_size = len(svm_x_test)
	svm_x_test = np.array(svm_x_test).reshape(dataset_size,-1)
	svm_y_test = [np.where(r==1)[0][0] for r in svm_y_test]
	svm_y_test = np_utils.to_categorical(svm_y_test,4)
	svm_y_train = np_utils.to_categorical(svm_y_train,4)
	# train scaler
	# scaler = StandardScaler()
	# scaler.fit(svm_x_train)
	# pickle.dump(scaler,open('scaler.p','wb'))

	# retrieve scaler
	scaler  = pickle.load(open(filePath + '/SpeakAI_data/scaler.p','rb'))
	# print(scaler.mean_,scaler.var_)
	svm_x_train = scaler.transform(svm_x_train)
	svm_x_test = scaler.transform(svm_x_test)
	svm_x_train,svm_y_train = shuffle(svm_x_train,svm_y_train)
	print(svm_x_train.shape)
	pickle.dump(svm_x_train,open(filePath + '/SpeakAI_data/x_train.p','wb'))
	pickle.dump(svm_y_train,open(filePath + '/SpeakAI_data/y_train.p','wb'))
	pickle.dump(svm_x_test,open(filePath + '/SpeakAI_data/x_test.p','wb'))
	pickle.dump(svm_y_test,open(filePath + '/SpeakAI_data/y_test.p','wb'))

def dataPreload():
	global svm_x_train,svm_y_train,svm_x_test,svm_y_test
	svm_x_train = pickle.load(open(filePath + '/SpeakAI_data/x_train.p','rb'))
	svm_y_train = pickle.load(open(filePath + '/SpeakAI_data/y_train.p','rb'))
	svm_x_test = pickle.load(open(filePath + '/SpeakAI_data/x_test.p','rb'))
	svm_y_test = pickle.load(open(filePath + '/SpeakAI_data/y_test.p','rb'))
	return svm_x_train,svm_y_train,svm_y_test,svm_y_test
dataPreload()
# preload()

def main(testbool):
<<<<<<< HEAD
	global svm_x_train,svm_y_train,svm_x_test,svm_y_test,clf
=======
<<<<<<< HEAD
	global svm_x_train,svm_y_train,svm_x_test,svm_y_test,clf
=======
	global svm_x_train,svm_y_train,svm_x_test,svm_y_test
>>>>>>> db59719d3b7d9d0a822b133c0cab79e49eda9524
>>>>>>> d8b5f6b1a7026caa18e5a01ae12d89eb99b6ef15
	if testbool:
		model_test(svm_x_test,svm_y_test, svm_x_train,svm_y_train)
		
	else:	
		#print(svm_y_train.shape)
<<<<<<< HEAD
		model = dense()
		model_train(svm_x_train,svm_y_train,svm_x_test,svm_y_test ,model)
=======
<<<<<<< HEAD
		model = dense()
		model_train(svm_x_train,svm_y_train,svm_x_test,svm_y_test ,model)
=======
		model_train(svm_x_train,svm_y_train,svm_x_test,svm_y_test ,dense())
>>>>>>> db59719d3b7d9d0a822b133c0cab79e49eda9524
>>>>>>> d8b5f6b1a7026caa18e5a01ae12d89eb99b6ef15
	
if __name__ == '__main__':
	main()
