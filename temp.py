def hel():
	global clf,model2,x_test

	rootdir1 = './webData/speaker'
	testdata = test_data(rootdir1)
	x_test = testdata[0]
	svm_x_test = []

	# optimize better

	for i in range(len(x_test)):
		x_1 = np.expand_dims(x_test[i], axis=0)
		#x_1 = preprocess_input(x_1)
		flatten_2_features = model2.predict(x_1)
		svm_x_test.append(flatten_2_features)


	svm_x_test = np.array(svm_x_test)
	dataset_size = len(svm_x_test)
	svm_x_test = np.array(svm_x_test).reshape(dataset_size,-1)
	predicted  = clf.predict(svm_x_train)
	print(predicted)

def wav2png():
	
	files = os.listdir(r'''./webData/wavs''')
	
	if not os.path.exists(r'''./webData/speaker'''):
		os.makedirs(r'''./webData/speaker''')
	
	count = 1
	for f in files:
		cmdstring = 'sox "{}" -n spectrogram -r -o "{}"'.format(r'''./webData/wavs/{}'''.format( f), r'''./webData/speaker/{}.png'''.format( count))
		subprocess.call(cmdstring, shell=True)
		count = count + 1
	