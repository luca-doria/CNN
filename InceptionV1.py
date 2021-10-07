import keras
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm

from keras.models import Model
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, AveragePooling2D, Flatten

import math 
import numpy as np
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler

## Settings #############################################

events = 15000

px = 128
py = 128
clr = 1 

## Load training data ###################################
X_train = np.ndarray(shape=(events,px,py,clr))
Y_train = np.ndarray(shape=(events,3))

fq = open('Inception_training.dat','r')
fy = open('All_data_coordinates_training.dat','r')

## Load input matrices ##################################

l=0
ln=0
j=0
for line in fq.readlines():
     line = line.strip()
     col = line.split() #list

     i=0

     for cl in col:
         if (float(cl)>0.01):
             X_train[l][i][j][0] = float(cl) #j-i
         else:
             X_train[l][i][j][0] = 0 #j-i
         #print str(l) + " " + str(i) + " " + str(j)
         i = i+1

     ln = ln +1

     j = j+1

     if (ln%px==0):
         l = l+1
         j=0

print("Input matrices loaded..")

## Load outputs #########################################

i=0
l=0
for line in fy.readlines():
     line = line.strip()
     col = line.split()
     i=0
     for cl in col:
         Y_train[l][i] = (float(cl)/840.0 + 1)*0.5 
         i=i+1

     l=l+1

print("Coordinates loaded..")

## Prepare data #########################################

#reshape data to fit model
X_train = X_train.reshape(events,px,py,clr) #1 = number of "colors"
Y_train = Y_train.reshape(events, 3)

## Defining the inception module ##
def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name) # Concatenating the paths that the NN took and using the output as an input for
                                                                                       # further layers.
    
    return output

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value = 0.2)

## Full network as described in the paper https://arxiv.org/abs/1409.4842 ##

input_layer = Input(shape=(128, 128, 1)) ## The network was originally made for 224x224, they mention in the paper that this is the size of their receptive field
                                          # It doesn't compile 64x64 inputs, but I just checked and it seems to accept 80x80 - I was going with powers of two when checking. 
                                          # Maybe we could try this. The input files would then be much smaller.

# Going deep

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

# Going wide

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=192,
                     filters_5x5_reduce=32,
                     filters_5x5=96,
                     filters_pool_proj=64,
                     name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=192,
                     filters_3x3_reduce=96,
                     filters_3x3=208,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_pool_proj=64,
                     name='inception_4a')


x1 = AveragePooling2D((5, 5), strides=3)(x)
x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(3, activation='sigmoid', name='auxilliary_output_1')(x1)

x = inception_module(x,
                     filters_1x1=160,
                     filters_3x3_reduce=112,
                     filters_3x3=224,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4d')


x2 = AveragePooling2D((5, 5), strides=3)(x)
x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(3, activation='sigmoid', name='auxilliary_output_2')(x2)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=384,
                     filters_3x3_reduce=192,
                     filters_3x3=384,
                     filters_5x5_reduce=48,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

x = Dropout(0.4)(x)

x = Dense(3, activation='sigmoid', name='output')(x)

model = Model(input_layer, [x, x1, x2], name='inception_v1')

model.summary()

epochs = 300
initial_lrate = 0.005

def decay(epoch, steps=10):
    initial_lrate = 0.005
    drop = 0.97
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

lr_sc = LearningRateScheduler(decay, verbose=1)

model.compile(loss=['mse', 'mse', 'mse'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['mse']) # there are two auxiliary outputs in addition to final one in the architecture,
                                                                                                      # so there is a need for these three 'mse' in loss and loss_weights which apply
                                                                                                      # to each of the outputs. If I remember correctly this was done, so that the
                                                                                                      # NN layers in the middle don't die off as training progresses.

history = model.fit(X_train, [Y_train, Y_train, Y_train], validation_split=0.2, epochs=epochs, batch_size=32, callbacks=[lr_sc])

predictions = model.predict(X_train)

# Save model to JSON
model_json = model.to_json()
with open("model_inception.json", "w") as json_file:
     json_file.write(model_json)
# Save weights to HDF5
model.save_weights("model_inception.h5")
print("Saved model to disk")

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

plt.savefig('loss_inception.pdf')

######################### Testing starts here ############################

## Settings #############################################

events_test = 4999

px = 128 #matrix X pixels
py = 128 #matrix Y pixels
clr = 1 #number of "colors"

## Load training data ###################################

X_test = np.ndarray(shape=(events_test,px,py,clr))
Y_test = np.ndarray(shape=(events_test,3))

fq_test = open('Inception_test.dat','r')
fy_test = open('All_data_coordinates_test.dat','r')


## Load input matrices ##################################

l=0
ln=0
j=0
for line in fq_test.readlines():
     line = line.strip()
     col = line.split() #list

     i=0

     for cl in col:
         if (float(cl)>0.01):
             X_test[l][i][j][0] = float(cl) #j-i
         else:
             X_test[l][i][j][0] = 0 #j-i
         #print str(l) + " " + str(i) + " " + str(j)
         i = i+1

     ln = ln +1

     j = j+1

     if (ln%px==0):
         l = l+1
         j=0
         
print("Position test matrices loaded..")

i=0
l=0
for line in fy_test.readlines():
     line = line.strip()
     col = line.split() #list
     i=0
     for cl in col:
         Y_test[l][i] = (float(cl.replace(',','.'))/840.0 + 1)*0.5 #variable range =-840..+840 normalize to 0-1
         i=i+1

     l=l+1

print("Test coordinates loaded..")

X_test = X_test.reshape(events_test,px,py,clr)

## Load model ###########################################

# load json model description and create model
json_file = open('model_inception.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_inception.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
score = model.evaluate(X_test, [Y_test, Y_test, Y_test]) #verbose=0
print((model.metrics_names[0], score[0]))
predictions = model.predict(X_test)
predictions = np.array(predictions[0])

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.scatter(predictions[:,0],Y_test[:,0],s=2)

x_hist.hist(predictions[:,0], 60, histtype='stepfilled',
orientation='vertical')
x_hist.invert_yaxis()

y_hist.hist(Y_test[:,0], 60, histtype='stepfilled',
orientation='horizontal')
y_hist.invert_xaxis()

plt.savefig('x_test.pdf')

#plt.close()

##########

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.scatter(predictions[:,1],Y_test[:,1],s=2)

x_hist.hist(predictions[:,1], 60, histtype='stepfilled',
orientation='vertical')
x_hist.invert_yaxis()

y_hist.hist(Y_test[:,1], 60, histtype='stepfilled',
orientation='horizontal')
y_hist.invert_xaxis()

plt.savefig('y_test.pdf')

#plt.close()

##########

fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.1, wspace=0.1)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.scatter(predictions[:,2],Y_test[:,2],s=2)

x_hist.hist(predictions[:,2], 60, histtype='stepfilled',
orientation='vertical')
x_hist.invert_yaxis()

y_hist.hist(Y_test[:,2], 60, histtype='stepfilled',
orientation='horizontal')
y_hist.invert_xaxis()

plt.savefig('Results/z_test.pdf')

#plt.close()

###########

plt.close()
plt.cla()
plt.clf()

predictions[:,0] =  840*(2*predictions[:,0] - 1)
Y_test[:,0]     =  840*(2*Y_test[:,0] - 1)
xd = predictions[:,0] - Y_test[:,0]

predictions[:,1] =  840*(2*predictions[:,1] - 1)
Y_test[:,1]     =  840*(2*Y_test[:,1] - 1)
yd = predictions[:,1] - Y_test[:,1]

predictions[:,2] =  840*(2*predictions[:,2] - 1)
Y_test[:,2]     =  840*(2*Y_test[:,2] - 1)
zd = predictions[:,2] - Y_test[:,2]

plt.subplot(3,1,1)
n_x, bins_x, patches = plt.hist( xd,
200,histtype='stepfilled',range=[-200,200], density=True)
(mu, sigma) = norm.fit(xd)
y = norm.pdf(bins_x, mu, sigma)
l = plt.plot(bins_x, y, 'r--', linewidth=2)
print("Fit results: mu = %.2f,  sigma = %.2f" % (mu, sigma))

plt.subplot(3,1,2)
n_y, bins_y, patches = plt.hist( yd,
200,histtype='stepfilled',range=[-200,200], density=True)
mu, sigma = norm.fit(yd)
y = norm.pdf(bins_y, mu, sigma)
l = plt.plot(bins_y, y, 'r--', linewidth=2)
print("Fit results: mu = %.2f,  sigma = %.2f" % (mu, sigma))

plt.subplot(3,1,3)
n_z, bins_z, patches = plt.hist( zd,
200,histtype='stepfilled',range=[-200,200], density=True)
mu, sigma = norm.fit(zd)
y = norm.pdf(bins_z, mu, sigma)
l = plt.plot(bins_z, y, 'r--', linewidth=2)
print("Fit results: mu = %.2f,  sigma = %.2f" % (mu, sigma))

plt.savefig('diff_test.pdf')

