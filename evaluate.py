import sys, os
import argparse
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, hamming_loss
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from training_functions import allow_growth_for_gpu

# default data folder
default_args_data = './data/spectrograms-test'
default_args_models = './models'
default_args_model_name = 'model'

# script
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default=default_args_data,
    help='Data folder to evaluate on. Default: {}'.format(default_args_data))
parser.add_argument('-m', '--models', default=default_args_models,
    help='Models folder. Default: {}'.format(default_args_models))
parser.add_argument('-n', '--name', default=default_args_model_name,
    help='Model name, without ".h5". Default: {}'.format(default_args_model_name))
args = parser.parse_args()

# load model:
allow_growth_for_gpu()
model_path = os.path.join(args.models, args.name) + '.h5'
if not os.path.isfile(model_path):
    raise FileNotFoundError('Model file {} not found'.format(model_path))
model = load_model(model_path)

# get data:
img_width, img_height = 128, 128
batch_size = 32
datagen = ImageDataGenerator(rescale=1./255)
validation_generator = datagen.flow_from_directory(
    args.data,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

num_batches = int(validation_generator.samples / batch_size)
class_names = list(validation_generator.class_indices.keys())

y_total = []
y_pred_total = []

for i in range(num_batches):
    [X, Y] = validation_generator.next()
    Y_pred = model.predict(X)
    # one hot encoding to class indices:
    y = np.argmax(Y, 1)
    y_pred = np.argmax(Y_pred, 1)
    y_total += list(y)
    y_pred_total += list(y_pred)

# plot confusion matrix:
conf_matrix = confusion_matrix(y_total, y_pred_total)

# normalize:
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots()
im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...s
ax.set(xticks=np.arange(conf_matrix.shape[1]),
    yticks=np.arange(conf_matrix.shape[0]),
    # ... and label them with the respective list entries
    xticklabels=class_names, yticklabels=class_names,
    title='Confusion matrix',
    ylabel='True label',
    xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(j, i, format(conf_matrix[i, j], fmt),
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > thresh else "black")
fig.tight_layout()

plt.show()

# compute Hamming loss:
print('Hamming loss: {}'.format(hamming_loss(y_total, y_pred_total))) 