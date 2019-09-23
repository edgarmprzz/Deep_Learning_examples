import numpy as np
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

weights_save_name = "weights.hdf5"
CONTINUE_TRAINING_WHERE_YOU_LEFT_OFF=False
full_dataset_filename = 'EEG_driving_data_sample.mat'
features_filename = 'extracted_features.mat'
input_features = 89

dataset = None
model = None
model_generated = False

def load_dataset(filename):
    print sio.whosmat(filename)
    data = sio.loadmat(filename)
    return data

def format_dataset(data):
    input_data = np.concatenate((np.swapaxes(data['extracted_features_class_0'],0,1),np.swapaxes(data['extracted_features_class_1'],0,1)),axis=0)
    out_data = np.concatenate((np.zeros(data['extracted_features_class_0'].shape[1],dtype=int),np.ones(data['extracted_features_class_1'].shape[1],dtype=int)),axis=0)
    output_data = out_data
    # Convert to one-hot encoding
    output_data = np.zeros((len(out_data),2),dtype=float)
    output_data[np.arange(len(out_data)),out_data] = 1
    # Shuffle data
    '''
    print input_data[0:5,:],output_data[0:5]
    input_data, output_data = shuffle(input_data, output_data, random_state=0)
    input_data, output_data = shuffle(input_data, output_data, random_state=35)
    input_data, output_data = shuffle(input_data, output_data, random_state=83)
    input_data, output_data = shuffle(input_data, output_data, random_state=34)
    input_data, output_data = shuffle(input_data, output_data, random_state=9)
    input_data, output_data = shuffle(input_data, output_data, random_state=56)
    input_data, output_data = shuffle(input_data, output_data, random_state=73)
    input_data, output_data = shuffle(input_data, output_data, random_state=19)
    input_data, output_data = shuffle(input_data, output_data, random_state=99)
    print input_data[0:5,:],output_data[0:5]
    '''
    n_2 = len(output_data)/2
    for i in range(n_2):
        if (i%2==0):
            output_data[i],output_data[i+n_2] = output_data[i+n_2],output_data[i]
            input_data[i],input_data[i+n_2] = input_data[i+n_2],input_data[i]
    input_data = normalize(input_data, norm='l2')
    return input_data, output_data

def format_raw_dataset(data):
    c_0 = data['data_class_0']
    c_0 = np.reshape(c_0,(c_0.shape[0]*c_0.shape[1],c_0.shape[2]))
    c_1 = data['data_class_1']
    c_1 = np.reshape(c_1,(c_1.shape[0]*c_1.shape[1],c_1.shape[2]))
    input_data = np.concatenate((np.swapaxes(c_0,0,1),np.swapaxes(c_1,0,1)),axis=0)
    out_data = np.concatenate((np.zeros(c_0.shape[1],dtype=int),np.ones(c_1.shape[1],dtype=int)),axis=0)
    output_data = out_data
    # Convert to one-hot encoding
    #output_data = np.zeros((len(out_data),2),dtype=float)
    #output_data[np.arange(len(out_data)),out_data] = 1
    # Shuffle data
    print input_data[0:5,:],output_data[0:5]
    input_data, output_data = shuffle(input_data, output_data, random_state=0)
    input_data, output_data = shuffle(input_data, output_data, random_state=35)
    input_data, output_data = shuffle(input_data, output_data, random_state=83)
    print input_data[0:5,:],output_data[0:5]
    return input_data, output_data

def load_pretrained_weights():
    global model
    try:
        model.load_weights(weights_save_name)
    except:
        print('Pre-trained weights do not exist. Please train model to obtain weights')

# Graph the model. Edit the model here if desired
def generate_model(load_weights=False):
    global model
    model = Sequential()
    model.add(Dense(512, input_dim=input_features, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.3))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    if(load_weights==True): load_pretrained_weights()
    model_generated = True

def train(x,y,batch,epoch,split_percent,load_weights=False):
    global model
    if(model_generated==False):
        generate_model(load_weights=load_weights)
    elif(load_weights==True):
        model.load_weights(weights_save_name)
    callbacks_list = [ModelCheckpoint(weights_save_name, monitor='loss', verbose=0, save_best_only=True, mode='auto', save_weights_only='True')]
    model.fit(x, y, batch_size=batch, epochs=epoch, verbose=0, callbacks=callbacks_list, shuffle=False, validation_split=split_percent)
    #model.fit(x, y, batch_size=batch, epochs=epoch, verbose=2, shuffle=False, validation_split=split_percent)

def train_in_steps(x, y,batch,epoch,split_percent,load_weights=False):
    global model
    if(model_generated==False):
        generate_model(load_weights=load_weights)
    elif(load_weights==True):
        model.load_weights(weights_save_name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_percent, random_state=42, shuffle=True)
    for i in range(epoch):
        print model.train_on_batch(x_train, y_train), model.test_on_batch(x_test, y_test)


def plot_data(x,y):
    plt.plot(x)
    plt.show()

def test_percentage(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
    ones = 0
    zeros = 0
    for i in y_train:
        if i == 1:
            ones += 1
        else:
            zeros += 1
    print ones,zeros
    ones = 0
    zeros = 0
    for i in y_test:
        if i == 1:
            ones += 1
        else:
            zeros += 1
    print ones,zeros

dataset = load_dataset(features_filename)
x_data, y_data = format_dataset(dataset)
#dataset = load_dataset(full_dataset_filename)
#x_data, y_data = format_raw_dataset(dataset)
#plot_data(x_data, y_data)
#test_percentage(x_data, y_data)
#train(x_data, y_data, 1600, 100000, 0.2, CONTINUE_TRAINING_WHERE_YOU_LEFT_OFF)
train_in_steps(x_data, y_data, 1600, 10000, 0.2, CONTINUE_TRAINING_WHERE_YOU_LEFT_OFF)


'''
# Graph the model. Edit the model here if desired
# 62% validation accuracy
def generate_model(load_weights=False):
    global model
    model = Sequential()
    model.add(Dense(256, input_dim=input_features, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.add(Dense(250, activation='relu'))
    #model.add(Dropout(0.3))
    #model.add(Dense(20, activation='tanh'))
    #model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    if(load_weights==True): load_pretrained_weights()
    model_generated = True
'''
