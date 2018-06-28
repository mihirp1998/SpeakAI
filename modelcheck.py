import keras
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
class Histories(keras.callbacks.Callback):

	def __init__(self, filepath, monitor='val_loss', verbose=0,
				 save_best_only=False, save_weights_only=False,
				 mode='auto', period=1):
		super(Histories, self).__init__()
		self.monitor = monitor
		self.verbose = verbose
		self.filepath = filepath
		self.save_best_only = save_best_only
		self.save_weights_only = save_weights_only
		self.period = period
		self.epochs_since_last_save = 0

		if mode not in ['auto', 'min', 'max']:
			warnings.warn('ModelCheckpoint mode %s is unknown, '
						  'fallback to auto mode.' % (mode),
						  RuntimeWarning)
			mode = 'auto'

		if mode == 'min':
			self.monitor_op = np.less
			self.best = np.Inf
		elif mode == 'max':
			self.monitor_op = np.greater
			self.best = -np.Inf
		else:
			if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
				self.monitor_op = np.greater
				self.best = -np.Inf
			else:
				self.monitor_op = np.less
				self.best = np.Inf



	def on_train_begin(self, logs={}):
		self.aucs = []
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		logs = logs or {}
		self.epochs_since_last_save += 1
		if self.epochs_since_last_save >= self.period:
			self.epochs_since_last_save = 0
			filepath = self.filepath.format(epoch=epoch + 1, **logs)
			if self.save_best_only:
				current = logs.get('val_categorical_accuracy')
<<<<<<< HEAD
				# print('this is current',current,logs)
=======
				print('this is current',current,logs)
>>>>>>> d8b5f6b1a7026caa18e5a01ae12d89eb99b6ef15
				if current is None:
					warnings.warn('Can save best model only with %s available, '
								  'skipping.' % (self.monitor), RuntimeWarning)
				else:
					if self.monitor_op(current, self.best):
<<<<<<< HEAD
						open('trainDetails.txt','wb').write('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
								  ' saving model to %s'
								  % (epoch + 1, self.monitor, self.best,
									 current, filepath))
=======
>>>>>>> d8b5f6b1a7026caa18e5a01ae12d89eb99b6ef15
						if self.verbose > 0:
							print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
								  ' saving model to %s'
								  % (epoch + 1, self.monitor, self.best,
									 current, filepath))
						self.best = current
						if self.save_weights_only:
							self.model.save_weights(filepath, overwrite=True)
						else:
							self.model.save(filepath, overwrite=True)
					else:
						if self.verbose > 0:
							print('\nEpoch %05d: %s did not improve from %0.5f' %
								  (epoch + 1, self.monitor, self.best))
			else:
				if self.verbose > 0:
					print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
				if self.save_weights_only:
					self.model.save_weights(filepath, overwrite=True)
				else:
					self.model.save(filepath, overwrite=True)

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return