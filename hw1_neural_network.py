import numpy as np
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# Constants
weights_save_name = "neural_network_weights.hdf5"
CONTINUE_TRAINING_WHERE_YOU_LEFT_OFF=True
features_filename = 'extracted_features.mat'
input_features = 60
run_training = True
run_testing = True
run_ROC = True

# Hyper Parameters
learning_rate = 0.001
training_epochs = 1000
batch_size = 1400

# Global variables
dataset = None
model = None
model_generated = False
ROC_analysis = []

def load_dataset(filename):
    print sio.whosmat(filename)
    data = sio.loadmat(filename)
    return data

def format_dataset(data):
    input_data = np.concatenate((np.swapaxes(data['extracted_features_class_0'],0,1),np.swapaxes(data['extracted_features_class_1'],0,1)),axis=0)
    output_data = np.concatenate((np.zeros(data['extracted_features_class_0'].shape[1],dtype=int),np.ones(data['extracted_features_class_1'].shape[1],dtype=int)),axis=0)
    # Normalize data
    input_data = normalize(input_data, norm='l2')
    # Shuffle data
    input_data, output_data = shuffle(input_data, output_data, random_state=0)
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
    model.add(Dense(1000, input_dim=input_features, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if(load_weights==True): load_pretrained_weights()
    model_generated = True

def train(x,y,batch,epoch,split_percent,load_weights=False):
    global model
    if(model_generated==False):
        generate_model(load_weights=load_weights)
    elif(load_weights==True):
        model.load_weights(weights_save_name)
    callbacks_list = [ModelCheckpoint(weights_save_name, monitor='loss', verbose=1, save_best_only=True, mode='auto', save_weights_only='True')]
    model.fit(x, y, batch_size=batch, epochs=epoch, verbose=2, callbacks=callbacks_list, shuffle=False, validation_split=split_percent)

def predict(x):
    global model
    if(model_generated==False):
        generate_model(load_weights=True)
    else:
        model.load_weights(weights_save_name)
    return model.predict(x,verbose=1)

# Consider 1 as positive and 0 as negetive
def ROC_AUC_analysis(pred_test, y_eval):
    sensitivity = []
    specificity = []
    for thresh in np.linspace(1,0,100):
        true_positives = 0
        false_positives = 0
        true_negetive = 0
        false_negetive = 0
        for i in range(len(pred_test)):
            predicted_class = -1
            if pred_test[i]>thresh:
                predicted_class = 1
            else:
                predicted_class = 0
            if((predicted_class == 1) & (y_eval[i] == 1)):
                true_positives += 1
            if((predicted_class == 1) & (y_eval[i] == 0)):
                false_positives += 1
            if((predicted_class == 0) & (y_eval[i] == 1)):
                false_negetive += 1
            if((predicted_class == 0) & (y_eval[i] == 0)):
                true_negetive += 1
        sensitivity.append(float(true_positives)/(true_positives+false_negetive))
        specificity.append(1-(float(true_negetive)/(true_negetive+false_positives)))
    AUC = np.trapz(sensitivity,x=specificity)
    return {'sensitivity':sensitivity,'specificity':specificity,'AUC':AUC}

def plot_ROC_AUC():
    avg_auc = 0
    n=0
    for ROC in ROC_analysis:
        print 'Area Under Curve (AUC) is: ', ROC['AUC']
        plt.plot(ROC['specificity'], ROC['sensitivity'], label="AUC:"+str(ROC['AUC']),linewidth=4.0)
        avg_auc += ROC['AUC']
        n += 1
    print 'Average AUC: ', float(avg_auc)/n
    plt.ylabel('True positive rate (Sensitivity)', fontsize=28)
    plt.xlabel('False positive rate (1-Specificity)', fontsize=28)
    plt.title('ROC curve for binary classification using a neural network model', fontsize=38)
    plt.legend(loc=4)
    plt.show()

def plot_data(x,y):
    plt.plot(x)
    plt.show()

dataset = load_dataset(features_filename)
x_data, y_data = format_dataset(dataset)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
if run_training==True:
    train(x_train, y_train, batch_size, training_epochs, 0, CONTINUE_TRAINING_WHERE_YOU_LEFT_OFF)
if run_testing==True:
    predictions = predict(x_test)
    if run_ROC==True:
        ROC_analysis.append(ROC_AUC_analysis(predictions,y_test))
        plot_ROC_AUC()
