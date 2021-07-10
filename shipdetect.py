# %% [code]
import os

os.listdir("../input/ships-in-satellite-imagery/shipsnet")


# %% [code]
import numpy as np
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
list

###########################################################################
## DATA LOADING
###########################################################################
with open('../input/ships-in-satellite-imagery/shipsnet.json') as jsFile:
    data = json.load(jsFile)

# print("foo")
filename = "json.sav"
with open(filename, "wb") as df:
    pickle.dump(data, df)

with open(filename, "rb") as df:
    data = pickle.load(df)
print("la")
# # Extract the Data and the labels from the json
compressedImages = data["data"]
labels = data["labels"]


###########################################################################
## DATA PADDING AND EXTRACTION
###########################################################################

# # print(len(compressedImages))
noImages = len(compressedImages)
images = np.asarray(compressedImages)
images = np.reshape(images, (noImages, 3, 80, 80))
labels = np.asarray(labels)

labelsFile = "labels.sav"
with open(labelsFile, "wb") as labf:
    labels = to_categorical(labels, num_classes=2)
    pickle.dump(labels, labf) 

imagesPadded = np.pad(images, ((0, 0), (0, 0), (24, 24), (24, 24)), "constant", constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
imagesPaddedFile = "imagesPadded.sav"

with open(imagesPaddedFile, "wb") as impf:
    pickle.dump(imagesPadded, impf) 

with open(imagesPaddedFile, "rb") as impf:
    imagesPadded = pickle.load(impf)
    
with open(labelsFile, "rb") as labf:
    labels = pickle.load(labf)

# ###########################################################################
# ## DATASET GENERATION
# ###########################################################################

x_train, x_test, y_train, y_test = train_test_split(imagesPadded, labels, shuffle=True, random_state=77)

with open("x_ship_train.sav", "wb") as x_trainf:
    pickle.dump(x_train, x_trainf)

with open("x_ship_test.sav", "wb") as x_testf:
    pickle.dump(x_test, x_testf)

with open("y_ship_train.sav", "wb") as y_trainf:
    pickle.dump(y_train, y_trainf)

with open("y_ship_test.sav", "wb") as y_testf:
     pickle.dump(y_test, y_testf)
image = image.array_to_img(imagesPadded[3000], data_format="channels_first")
print(labels[3000])
print(imagesPadded.shape)

# # plt.figure()
# # plt.imshow(image)
# # plt.show()


###########################################################################
## TRAIN DATA AUGMENTATION
##########################################################################

with open("x_ship_train.sav", "rb") as x_trainf:
    x_train = pickle.load(x_trainf)

with open("y_ship_train.sav", "rb") as y_trainf:
    y_train = pickle.load(y_trainf)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
datagen_args = dict(
    data_format="channels_first",
    rotation_range=360,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range = [0.1, 0.2],
    # brightness_range=(0.1, 0.9),
    horizontal_flip=True, fill_mode="constant", 
    cval=0)

image_datagen = image.ImageDataGenerator(**datagen_args)
label_datagen = image.ImageDataGenerator(**datagen_args)

seed = 77
#image_datagen.fit(x_train, augment=True, seed=seed)
#label_datagen.fit(y_train, augment=True, seed=seed)
#it = image_datagen.flow(x_train, y_train, seed=seed)
#print(it.n)

# %% [code]
y_test.shape





# %% [code]
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model

import joblib

gpu_devices = tf.config.list_physical_devices('GPU')
for gpu in gpu_devices:
   tf.config.experimental.set_memory_growth(gpu, True)

num_epochs = 40
num_classes = 2
batch_size = 128

x_train = joblib.load('x_ship_train.sav')
y_train = joblib.load('y_ship_train.sav')
x_test = joblib.load('x_ship_test.sav')
y_test = joblib.load('y_ship_test.sav')

input_shape = (3, 128, 128)

# Create Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape, data_format='channels_first', padding="same", name="conv2d_1"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first', name="max_pooling2d_1"))
model.add(Dropout(0.1))
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu', data_format='channels_first', padding="same", name="conv2d_2"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first', name="max_pooling2d_2"))
model.add(Dropout(0.1))
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu', data_format='channels_first', padding="same", name="conv2d_3"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first', name="max_pooling2d_3"))
model.add(Dropout(0.1))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', data_format='channels_first', padding="same", name="conv2d_4"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first', name="max_pooling2d_4"))
model.add(Dropout(0.1))

model.add(Flatten(data_format='channels_first'))
model.add(Dense(24, activation='relu', name="dense_1"))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax', name="dense_2"))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer="adam",
              metrics=['accuracy'])


weights_file = "ship_weights.hdf5"
checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True,monitor='val_accuracy')


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_data=(x_test, y_test), callbacks=[checkpointer])

model.load_weights(weights_file)

model_filename = "ship_detection_model.h5"
model.save(model_filename)

plot_model(model, to_file='ship.png', show_shapes=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% [code]
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# %% [code]
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
tf.compat.v1.disable_eager_execution()

# Clear any previous session.
tf.keras.backend.clear_session()

#save_pb_dir = './model'
model_fname = '../input/shiph5/ship_detection_model(1).h5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

model = load_model(model_fname)

session = tf.compat.v1.keras.backend.get_session()

input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]

# Prints input and output nodes names, take notes of them.
print(input_names, output_names)

frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)

# %% [code]
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

# %% [code]
graph_io.write_graph(trt_graph, "./model/",
                     "trt_graph.pb", as_text=False)

# %% [code]
!ls model -alh

# %% [code]
model.summary()


# %% [code]
os.listdir('../input/shiph5')

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]
x = np.expand_dims(x_test[0], axis=0)
x.shape


# %% [code]
import time
times = []

for i in range(20):
    start_time = time.time()
    preds = model.predict(x)
    delta = (time.time() - start_time)
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1/mean_delta
print('average(sec):{},fps:{}'.format(mean_delta,fps))

# Clear any previous session.
tf.keras.backend.clear_session()

# %% [code]


# %% [code]


# %% [code]


# %% [code]

