import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize

# Constants
weights_save_name = "logistic_regression_weights.ckpt"
CONTINUE_TRAINING_WHERE_YOU_LEFT_OFF=False
features_filename = 'extracted_features.mat'
input_features = 60
train_run = True
test_run = True
plot_ROC = True

# Hyper Parameters
learning_rate = 0.001
training_epochs = 7000
batch_size = 1400
display_step = 50
threshold = 0.5
n_folds = 5

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

def train_test_n_fold_split(x,y,folds):
    section_length = len(y)/folds
    for i in range(folds):
        x_test = x[i*section_length:(i+1)*section_length]
        x_train = np.concatenate((x[0:i*section_length],x[(i+1)*section_length:len(y)]),axis=0)
        y_test = y[i*section_length:(i+1)*section_length]
        y_train = np.concatenate((y[0:i*section_length],y[(i+1)*section_length:len(y)]),axis=0)
        yield x_train, x_test, y_train, y_test

def evalutate(session, x_eval, y_eval):
    pred_test = session.run(pred, feed_dict={x: x_eval, y: y_eval})
    correct = 0
    for i in range(len(pred_test)):
        predicted_class = -1
        if pred_test[i]>threshold:
            predicted_class = 1
        else:
            predicted_class = 0
        if predicted_class == y_test[i]:
            correct += 1
    return float(correct)/len(pred_test)

# Consider 1 as positive and 0 as negetive
def ROC_AUC_analysis(session, x_eval, y_eval):
    pred_test = session.run(pred, feed_dict={x: x_eval, y: y_eval})
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
    plt.title('ROC curve for binary classification using a logistic regression model (5-fold cross validation)', fontsize=38)
    plt.legend(loc=4)
    plt.show()

# Generate graph for the linear regression model
x = tf.placeholder(tf.float32, [None,input_features])
y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.random_normal([input_features,1]))
b = tf.Variable(tf.zeros([1]))
z = tf.matmul(x,W) + b
pred = tf.nn.sigmoid(z)
loss = y*tf.log(pred) + (1-y)*tf.log(1-pred)
cost = tf.reduce_mean(-tf.reduce_sum(loss))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Dataset
dataset = load_dataset(features_filename)
x_data, y_data = format_dataset(dataset)
y_data = np.reshape(y_data,(y_data.shape[0],1))

with tf.Session() as sess:
    # n fold training and validation
    for x_train, x_test, y_train, y_test in train_test_n_fold_split(x_data, y_data, n_folds):
        if CONTINUE_TRAINING_WHERE_YOU_LEFT_OFF == True:
            saver.restore(sess, weights_save_name)
            print("Model restored.")
        else:
            sess.run(init)

        if(train_run == True):
            for epoch in range(training_epochs):
                _, c, new_w,new_b,p,l,cl = sess.run([optimizer, cost, W,b,pred,loss,y], feed_dict={x: x_train, y: y_train})
                # Display logs per epoch step
                if (epoch+1) % display_step == 0:
                    acc = evalutate(sess,x_test,y_test)
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "Validation Accuracy=","{:.2f}".format(acc))

            print("Optimization Finished!")
            save_path = saver.save(sess, weights_save_name)
            print("Model saved in file: %s" % save_path)

        if(test_run == True):
            print 'Accuracy is: ', evalutate(sess,x_test,y_test)

        if(plot_ROC == True):
            ROC_analysis.append(ROC_AUC_analysis(sess,x_test,y_test))

    if(plot_ROC == True):
        plot_ROC_AUC()
