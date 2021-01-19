import dadrah.selection.quantile_regression as qure
import numpy as np
import tensorflow as tf
import setGPU


# *************************************
#				params
# *************************************

batch_sz = 32
quantile = 0.1
epochs_n = 10

# *************************************
#				data
# *************************************

N = 10000
x = np.random.uniform(low=0, high=10, size=(N,1)).astype('float32') # x in [0,10]
y = np.random.uniform(low=0, high=50, size=N).astype('float32') # y in [0,50]
train_dataset = tf.data.Dataset.from_tensor_slices((x[:-1000], y[:-1000])).shuffle(buffer_size=1024).batch(batch_sz)
valid_dataset = tf.data.Dataset.from_tensor_slices((x[-1000:], y[-1000:])).shuffle(buffer_size=1024).batch(batch_sz)


# *************************************
#				model
# *************************************

QR = qure.QuantileRegressionV2(n_layers=2, n_nodes=10)
QR_model = QR.make_model()
loss_fun = QR.make_quantile_loss(quantile)
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)


# *************************************
#				metrics
# *************************************



# *************************************
#				forward step
# *************************************


# *** training

@tf.function
def training_step(x_batch, y_batch):
	with tf.GradientTape() as tape:
		predictions = QR_model(x_batch, training=True)
		loss_value = loss_fun(y_batch, predictions)
	grads = tape.gradient(loss_value, QR_model.trainable_weights)
	optimizer.apply_gradients(zip(grads, QR_model.trainable_weights))
	return loss_value


def training_epoch(train_dataset):
	# run training step for each batch
	loss_per_epoch = 0
	for step, (x_batch, y_batch) in enumerate(train_dataset):
		loss_value = training_step(x_batch, y_batch)
		if step % 100 == 0:
			print("Training loss (for one batch) at step %d: %.4f" % (step, float(tf.math.reduce_sum(loss_value))))

		loss_per_epoch += float(tf.math.reduce_sum(loss_value))
	return loss_per_epoch / step


# *** validation

@tf.function
def validation_step(x_batch, y_batch):
	predictions = QR_model(x_batch)
	loss_value = loss_fun(y_batch, predictions)
	return loss_value

def validation_epoch(valid_dataset):
	# run training step for each batch
	loss_per_epoch = 0
	for step, (x_batch, y_batch) in enumerate(valid_dataset):
		loss_value = validation_step(x_batch, y_batch)
		loss_per_epoch += float(tf.math.reduce_sum(loss_value))
	return loss_per_epoch / step


# *************************************
#				train
# *************************************

# for each epoch
for epoch in range(epochs_n):
	loss_per_epoch_train = training_epoch(train_dataset)
	loss_per_epoch_valid = validation_epoch(valid_dataset)
	print('epoch {} loss training: {}, loss validation: {}'.format(epoch, loss_per_epoch_train/batch_sz, loss_per_epoch_valid/batch_sz))
