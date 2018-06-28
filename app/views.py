import keras
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
import pickle
from app import app
import warnings
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
from flask import render_template,request,send_from_directory
from werkzeug import secure_filename
from pydub import AudioSegment
import os
from os import listdir
from os.path import isfile, join
import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import subprocess

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

def model():
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
	model.load_weights('../SpeakAI_data/my_model_weights.h5')
	
	return model



def mp3ToWav():
	AudioSegment.converter = r"ffmpeg"#r"/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/ffmpeg"
#change the range(10)
	for i in range(10):
		files = os.listdir(r'''./data/train/audio{}'''.format(i))
		
		if not os.path.exists(r'''./data/train/wav{}'''.format(i)):
			os.makedirs(r'''/./data/train/wav{}'''.format(i))
		
		count = 1
		limit = 0
		for f in files:
			if(limit == 720): break
			sound = AudioSegment.from_mp3(r'''./data/train/audio{}/{}'''.format(i, f))
			out_f = sound.export(r'''./data/train/wav{}/{}.wav'''.format(i, count), format="wav")
			out_f.close()
			count = count + 1
			limit = limit + 1

def trimWav():
#change the range(10)
	for i in range(10):
		in_path = os.listdir('./data/train/wav{}'.format(i))
		if not os.path.exists(r'''./data/train/wavs{}'''.format(i)):
			os.makedirs(r'''./data/train/wavs{}'''.format(i))    
		for z in in_path:
			r = './data/train/wav{}/'.format(i)+z  
			out_path = './data/train/wavs{}/out%03d{}'.format(i,z)
			subprocess.call(['ffmpeg', '-i', r, '-f', 'segment','-segment_time', '5', '-c', 'copy', out_path])

def wav2png():
	
	files = os.listdir(r'''./webData/wavs''')
	
	if not os.path.exists(r'''./webData/speaker'''):
		os.makedirs(r'''./webData/speaker''')
	
	count = 1
	for f in files:
		cmdstring = 'sox "{}" -n spectrogram -r -o "{}"'.format(r'''./webData/wavs/{}'''.format( f), r'''./webData/speaker/{}.png'''.format( count))
		subprocess.call(cmdstring, shell=True)
		count = count + 1




# clf = pickle.load(open('clf.p','rb'))
model = model()
model2 = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
graph = tf.get_default_graph()

@app.route('/uploaderfor2', methods = [ 'POST'])
def upload_filefor2():
	print('helLO',request.files)

	f = request.files['wavFile']
	print('SA',f.filename)
	f.save('./webData/wavs/'+ secure_filename('input.wav'))
	# return 'file uploaded successfully'
	return 'done'

	# global headVal,bodyVal,m
	# global check
	return 'done'
	# if request.method == 'POST':
	# 	try:
	# 		print('get printed')
	# 		f = request.files['file']
	# 		print('SA',f.filename)
	# 		f.save('./webData/wavs/'+ secure_filename('input.wav'))
	# 		# return 'file uploaded successfully'
	# 		return render_template('index.html',article='<p>File Uploaded</p>')
	# 	except Exception:
	# 		check = False
	# 		return render_template('index.html',article='<p>Reupload! File Not Uploaded</p>')
	# else:
	# 	return render_template('result.html',head=headVal,body = bodyVal,final = m)	















@app.route('/uploader', methods = [ 'POST'])
def upload_file():
	print('helLO',request.method)
	global headVal,bodyVal,m
	global check
	if request.method == 'POST':
		try:
			print('get printed')
			f = request.files['file']
			print('SA',f.filename)
			f.save('./webData/wavs/'+ secure_filename('input.wav'))
			# return 'file uploaded successfully'
			return render_template('index.html',article='<p>File Uploaded</p>')
		except Exception:
			check = False
			return render_template('index.html',article='<p>Reupload! File Not Uploaded</p>')
	else:
		return render_template('result.html',head=headVal,body = bodyVal,final = m)	


@app.route('/open', methods = [ 'POST'])
def sendOutput():
	global clf,model2,x_test
	global graph
	wav2png()
	rootdir1 = './webData/speaker'
	testdata = test_data(rootdir1)
	print(testdata)
	x_test = testdata[0]
	svm_x_test = []

	# optimize better

	for i in range(len(x_test)):
		x_1 = np.expand_dims(x_test[i], axis=0)
		#x_1 = preprocess_input(x_1)
		with graph.as_default():
			flatten_2_features = model2.predict(x_1)
		svm_x_test.append(flatten_2_features)


	svm_x_test = np.array(svm_x_test)
	dataset_size = len(svm_x_test)
	svm_x_test = np.array(svm_x_test).reshape(dataset_size,-1)
	predicted  = clf.predict(svm_x_test)
	print(predicted,clf.predict_proba(svm_x_test))


	return render_template('result.html')





def nocache(view):
	@wraps(view)
	def no_cache(*args, **kwargs):
		response = make_response(view(*args, **kwargs))
		response.headers['Last-Modified'] = datetime.now()
		response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
		response.headers['Pragma'] = 'no-cache'
		response.headers['Expires'] = '-1'
		return response
		
	return update_wrapper(no_cache, view)








@app.route('/mic128.png')
def sendMic():
	return send_from_directory('./static', 'mic128.png')

@app.route('/save.svg')
def sendSave():
	return send_from_directory('./static', 'save.svg')


@app.route('/audiodisplay.js')
def sendAudio():
	return render_template('audiodisplay.js')

@app.route('/recorder.js')
def sendRecorder():
	return render_template('recorder.js')

@app.route('/main.js')
def sendMain():
	return render_template('main.js')

@app.route('/recorderWorker.js')
def sendRecoderwork():
	return render_template('recorderWorker.js')

	
@app.route('/WebAudioScheduler.js')
def sendWebAudioScheduler():
	return render_template('WebAudioScheduler.js')





@app.route('/component.css')
# @nocache
def sendcssa3():
	return send_from_directory('./static', 'component.css')


@app.route('/demo.css')
def sendcssa4():
	return send_from_directory('./static', 'demo.css')


@app.route('/normalize.css')
def sendcssa1():
	return send_from_directory('./static', 'normalize.css')		

@app.route('/jquery-v1.min.js')
def sendjqeury():
	return render_template('jquery-3.2.1.min.js')

@app.route('/custom-file-input.js')
def sendjqeury1():
	return render_template('custom-file-input.js')	

@app.route('/bootstrap.min.css')
def sendcss1():
	return send_from_directory('./static', 'bootstrap.min.css')	
@app.route('/animate.css')
def sendcss2():
	return send_from_directory('./static', 'animate.css')	
@app.route('/select2.min.css')
def sendcss3():
	return send_from_directory('./static', 'select2.min.css')	
@app.route('/perfect-scrollbar.css')
def sendcss4():
	return send_from_directory('./static', 'perfect-scrollbar.css')	

@app.route('/style.css')
def sendercss7():
	return send_from_directory('./static', 'style.css')	
@app.route('/util.css')
def sendcss5():
	return send_from_directory('./static', 'util.css')	
@app.route('/main.css')
def sendcss6():
	return send_from_directory('./static', 'main.css')	


@app.route('/codropsicons.eot')
def sendFont():
	return send_from_directory('./static', 'codropsicons.eot')

@app.route('/codropsicons.svg')
def sendFont1():
	return send_from_directory('./static', "codropsicons.svg")

@app.route('/codropsicons.ttf')
def sendFont2():
	return send_from_directory('./static', "codropsicons.ttf")



@app.route('/codropsicons.woff')
def sendFont3():
	return send_from_directory('./static',  "codropsicons.woff")		
@app.route('/plot.png')
@nocache
def sendImage1():
	return send_from_directory('./templates',  "plot.png")
@app.route('/new.png')
@nocache
def sendImage2():
	return send_from_directory('./templates',  "new.png")	

@app.route('/logo.jpg')
def sendImage():
	return send_from_directory('./static',  "logo.png")	



	# fig.canvas.draw()
	# data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	# img = PIL.Image.fromarray(data)
	# img.save("out.jpg")
	# plt.show()
	# sess.close()

@app.route('/loggedin', methods=['POST'])
def logindex():
	print('rendered')
	global user,passw,headVal,bodyVal,m
	if request.method == 'POST':
		print('notworking')
		user = request.form['user']
		passw = request.form['pass']
		print(user,passw)
		if user=="admin" and passw=="helloworld":
			return render_template('index.html')
		else:
			return "<h2>Wrong Password</h2>"

	if request.method == 'GET':
		print('is it coming')
		return render_template('result.html',head=headVal,body = bodyVal,final = m)		


@app.route('/')
def index():
	print('rendered')
	return render_template('login.html')