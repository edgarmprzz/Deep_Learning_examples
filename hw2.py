import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import itertools
import os

# Hyper parameters
alpha = 0.001
batch_size = 200
max_epochs = 4000
validation_update_per_epoch = 100
# Parameters for finding best network architecture
max_hidden_layers = 3
max_neuron_per_layer = 700
check_every_n_neuron = 100
start_with_n_neuron = 100

# Network definition
nn_architechture = {
    'activation':'relu',
    'n_neurons_in_input_layer':784,
    'n_neurons_in_hidden_layers':[200,300],
    'n_neurons_in_ouput_layer':10,
    'dropout_in_each_layer':[1.0]*max_hidden_layers,
    'cost':'softmax_cross_entropy_logits',
    'optimizer':tf.train.AdamOptimizer
}

# Dataset object
mnist = input_data.read_data_sets("./", one_hot = True)

activations = {
    'relu':tf.nn.relu,
    'sigmoid':tf.nn.sigmoid,
    'tanh':tf.nn.tanh
}

costs = {
    # Logits
    'softmax_cross_entropy_logits':tf.nn.softmax_cross_entropy_with_logits,
    'sigmoid_cross_entropy_logits':tf.nn.sigmoid_cross_entropy_with_logits,
    'hinge_loss':tf.losses.hinge_loss,
    # Predictions
    #'mean_squared_error':tf.losses.mean_squared_error,
    #'log_loss':tf.losses.log_loss,
    #'huber_loss':tf.losses.huber_loss,
    #'cosine_distance':tf.losses.cosine_distance,
    #'absolute_difference':tf.losses.absolute_difference
}

directories = {
    'save_weights':'./saved_weights',
    'tensorboard_graph':'./graphs',
    'plots':'./plots'
}

# Graph generation
def generate_dnn(nn):
    # Input
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32,[None,nn['n_neurons_in_input_layer']],name='input_images')
    # Output
    with tf.name_scope('output'):
        y = tf.placeholder(tf.float32,[None,nn['n_neurons_in_ouput_layer']],name='ouput_labels')
    # DNN output(prediction)
    pred_y = x
    with tf.name_scope('hidden'):
        for n_hidden_layer in range(len(nn['n_neurons_in_hidden_layers'])):
            with tf.name_scope('layer_'+str(n_hidden_layer)):
                w = tf.Variable(tf.random_normal([pred_y.shape[1].value,nn['n_neurons_in_hidden_layers'][n_hidden_layer]]),name='weights')
                b = tf.Variable(tf.random_normal([nn['n_neurons_in_hidden_layers'][n_hidden_layer]]),name='biases')
                pred_y = tf.add(tf.matmul(pred_y,w),b)
                pred_y = activations[nn['activation']](pred_y)
                pred_y = tf.nn.dropout(pred_y, nn['dropout_in_each_layer'][n_hidden_layer])
    with tf.name_scope('prediction'):
        pred_y = tf.add(tf.matmul(pred_y,tf.Variable(tf.random_normal([pred_y.shape[1].value,nn['n_neurons_in_ouput_layer']]),name='weights')),tf.Variable(tf.random_normal([nn['n_neurons_in_ouput_layer']]),name='biases'))
    # Return the graph
    return x,pred_y,y

# Metrics
def define_metrics(labels,predictions,nn):
    cost,optimize,accuracy = None,None,None
    with tf.name_scope('optimizer'):
        if (nn['cost'] == 'softmax_cross_entropy_logits' or nn['cost'] == 'sigmoid_cross_entropy_logits' or nn['cost'] == 'hinge_loss' ):
            cost = tf.reduce_mean(costs[nn['cost']](labels=labels, logits=predictions))
        else:
            cost = tf.reduce_mean(costs[nn['cost']](labels=labels, predictions=predictions))
        optimize = nn['optimizer'](alpha).minimize(cost)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1)), tf.float32))
    return cost,optimize,accuracy

def check_create_directories(directories):
    for directory in directories:
        if not os.path.exists(directories[directory]):
            os.makedirs(directories[directory])

# Visualize the graph on tensorboard
def plot_graph(sess):
    writer = tf.summary.FileWriter(directories['tensorboard_graph'], sess.graph)
    writer.close()

def plot_learning(log,nn):
    fig, ax1 = plt.subplots()
    ax1.plot(log['epoch'],log['train_accuracy'],label='Training accuracy',color='c',linewidth=2.0)
    ax1.plot(log['epoch'],log['test_accuracy'],label='Testing accuracy',color='r',linewidth=2.0)
    ax1.set_xlabel('Epochs (Iterations)')
    ax1.set_ylabel('Accuracy')
    ax1.legend(bbox_to_anchor=(1, 0.5), loc=1)
    ax2 = ax1.twinx()
    ax2.plot(log['epoch'],log['train_loss'],label='Training loss',color='b',linewidth=2.0)
    ax2.plot(log['epoch'],log['test_loss'],label='Testing loss',color='g',linewidth=2.0)
    ax2.set_ylabel('Loss')
    ax2.legend(bbox_to_anchor=(1, 0.5), loc=4)
    fig.tight_layout()
    plt.savefig(directories['plots']+'/'+str(nn['n_neurons_in_hidden_layers'])+'.png')
    plt.pause(0.1)
    plt.close()

def save_weights(sess):
    saver = tf.train.Saver()
    save_path = saver.save(sess, directories['save_weights']+"/trained_model_dnn.ckpt")
    print("Model saved in file: %s" % save_path)

def run_training(nn_architechture, save_weight=False, plot=False, tensorboard_graph=False):
    training_log = {'train_accuracy':[],'test_accuracy':[],'train_loss':[],'test_loss':[],'epoch':[]}
    final_accuracy,final_loss = 0,0
    x,y_pred,y = generate_dnn(nn_architechture)
    cost,optimize,accuracy = define_metrics(y,y_pred,nn_architechture)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(max_epochs):
            # Training: Run network on training dataset
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimize, feed_dict={x: batch_xs, y: batch_ys})
            train_acc,train_loss = sess.run([accuracy, cost], {x: batch_xs, y: batch_ys})
            if(epoch % validation_update_per_epoch == 0):
                # Validation: Run network on validation dataset
                batch_xs, batch_ys = mnist.validation.images,mnist.validation.labels
                test_acc,test_loss = sess.run([accuracy, cost], {x: batch_xs, y: batch_ys})
                print 'Epoch:',epoch,' Accuracy:',test_acc*100,' Cost:',test_loss
                # Add to log
                training_log['train_accuracy'].append(train_acc)
                training_log['test_accuracy'].append(test_acc)
                training_log['train_loss'].append(train_loss)
                training_log['test_loss'].append(test_loss)
                training_log['epoch'].append(epoch)
        # Evaluation: Run network on evaluation dataset
        batch_xs, batch_ys = mnist.test.images,mnist.test.labels
        final_accuracy,final_loss = sess.run([accuracy, cost], {x: batch_xs, y: batch_ys})
        print 'Final accuracy for network:',final_accuracy*100,' Cost:',final_loss
        # Visualize the graph on tensorboard
        if tensorboard_graph==True:
            plot_graph(sess)
        # Plot learning curve
        if plot==True:
            plot_learning(training_log,nn_architechture)
        # Save the weights to disk.
        if save_weight==True:
            save_weights(sess)
    return final_accuracy,final_loss,training_log

def plot_accuracies(log,variable):
    accs = [x['acc'] for x in log]
    var = [x[variable] for x in log]
    acc_max_index = np.argmax(accs)
    y_pos = np.arange(len(var))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if variable == 'nn':
        text = 'Maximum accuracy at '+str(log[acc_max_index]['nn']) + ',  ' + '{0:.5f}'.format(log[acc_max_index]['acc'])
        ax.plot(y_pos, accs, linewidth=3)
        ax.annotate(text, xy=(acc_max_index, log[acc_max_index]['acc']), xytext=(len(var)/2,log[acc_max_index]['acc']+0.01),
                arrowprops=dict(facecolor='black'),
                )
        ax.set_xlabel('Network architectures')
        plt.tight_layout(pad=3)
    else:
        plt.tight_layout(pad=5)
        plt.bar(y_pos, accs, align='center', alpha=0.5)
        plt.xticks(y_pos, var, rotation=20)
    ax.set_ylabel('Accuracy')
    plt.savefig(directories['plots']+'/comparison_best_'+variable+'.png')
    plt.pause(0.1)
    plt.close()

def find_best_network_from_all_networks():
    log = []
    for l in range(max_hidden_layers):
        for nn in itertools.product(range(start_with_n_neuron,max_neuron_per_layer,check_every_n_neuron),repeat=l+1):
            nn_architechture['n_neurons_in_hidden_layers'] = nn
            print '\n\nAnalysing network ',nn
            a,ls,lg = run_training(nn_architechture,plot=True)
            log.append({'acc':a,'nn':nn_architechture['n_neurons_in_hidden_layers']})
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy with this layer obtained at:',log[acc_max_index]
    np.save(directories['plots']+'/log_nn',log)
    plot_accuracies(log,'nn')
    return log[acc_max_index]['nn']

def find_best_activation_function():
    log = []
    for activation in activations:
        nn_architechture['activation'] = activation
        print '\n\nAnalysing network with ',activation, ' activation'
        a,ls,lg = run_training(nn_architechture,plot=True,tensorboard_graph=True)
        log.append({'acc':a,'activation':nn_architechture['activation']})
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy obtained with activation function:',log[acc_max_index]
    np.save(directories['plots']+'/log_activation',log)
    plot_accuracies(log,'activation')
    return log[acc_max_index]['activation']

def find_best_dropouts():
    log = []
    for d in itertools.product([0.7,1], repeat=len(nn_architechture['n_neurons_in_hidden_layers'])):
        nn_architechture['dropout_in_each_layer'] = d
        print '\n\nAnalysing network with ',d, ' dropout'
        a,ls,lg = run_training(nn_architechture,plot=True,tensorboard_graph=True)
        log.append({'acc':a,'dropout':nn_architechture['dropout_in_each_layer']})
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy obtained with dropout configuration:',log[acc_max_index]
    np.save(directories['plots']+'/log_dropout',log)
    plot_accuracies(log,'dropout')
    return log[acc_max_index]['dropout']

def find_best_cost_function():
    log = []
    for cost in costs:
        nn_architechture['cost'] = cost
        print '\n\nAnalysing network with ',cost, ' cost function'
        a,ls,lg = run_training(nn_architechture,plot=True,tensorboard_graph=True)
        log.append({'acc':a,'cost':nn_architechture['cost']})
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy obtained with cost function: ',log[acc_max_index]
    np.save(directories['plots']+'/log_cost',log)
    plot_accuracies(log,'cost')
    return log[acc_max_index]['cost']

def plot_multi_learning(logs):
    fig, ax1 = plt.subplots()
    for log in logs:
        ax1.plot(log[0]['epoch'],log[0]['test_accuracy'],label='alpha:'+str(log[1]),linewidth=2.0)
    ax1.set_xlabel('Epochs (Iterations)')
    ax1.set_ylabel('Accuracy')
    ax1.legend(bbox_to_anchor=(1, 0.5), loc=1)
    fig.tight_layout()
    plt.savefig(directories['plots']+'/multi_learning_curve.png')
    plt.pause(0.1)
    plt.close()

def find_best_learning_rate():
    log = []
    logs = []
    global alpha 
    for a in [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]:
        alpha = a
        print '\n\nAnalysing network with ',alpha, ' learning rate'
        a,ls,lg = run_training(nn_architechture,plot=True,tensorboard_graph=True)
        log.append({'acc':a,'alpha':alpha})
        logs.append([lg,alpha])
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy obtained with learning rate: ',log[acc_max_index]
    np.save(directories['plots']+'/log_alpha',log)
    np.save(directories['plots']+'/logs_alpha',logs)
    plot_accuracies(log,'alpha')
    plot_multi_learning(logs)
    return log[acc_max_index]['alpha']

def run():
    #### Q1 ####
    #nn_architechture['n_neurons_in_hidden_layers'] = [500,500,500]
    run_training(nn_architechture,plot=True,tensorboard_graph=True)
    #### Q2 ####
    #nn = find_best_network_from_all_networks()
    #nn_architechture['n_neurons_in_hidden_layers'] = nn
    #### Q3 ####
    #act = find_best_activation_function()
    #nn_architechture['activation'] = act
    #### Q4 ####
    #d = find_best_dropouts()
    #nn_architechture['dropout_in_each_layer'] = d
    #### Q5 ####
    #c = find_best_cost_function()
    #nn_architechture['cost'] = c

nn_architechture['n_neurons_in_hidden_layers'] = [600,500,600]
nn_architechture['activation'] = 'relu'
nn_architechture['dropout'] = [1.0,1.0,1.0]
nn_architechture['cost'] = 'sigmoid_cross_entropy_logits'
nn_architechture['optimizer'] = tf.train.GradientDescentOptimizer
find_best_learning_rate()



'''
Notes:
- Cost functions obtained from https://www.tensorflow.org/api_docs/python/tf/losses
- Refered to https://github.com/tensorflow/tensorflow/blob/4af9be964eff70b9f27f605e0f5b2cb04a5d03cc/tensorflow/contrib/learn/python/learn/datasets/mnist.py to get a better idea of how the imported dataset object works
def find_best_network():
    check_create_directories(directories)
    best_arch = []
    best_acc = 0
    nn_architechture['n_neurons_in_hidden_layers'] = []
    for l in range(max_hidden_layers):
        log = []
        print '\n\nAnalysing layer ',l
        nn_architechture['n_neurons_in_hidden_layers'].append(0)
        for n_neurons in range(start_with_n_neuron,max_neuron_per_layer,check_every_n_neuron):
            nn_architechture['n_neurons_in_hidden_layers'][l] = n_neurons
            a,ls,lg = run_training(nn_architechture,plot=True)
            log.append({'acc':a,'nn':nn_architechture['n_neurons_in_hidden_layers']})
        acc_max_index = np.argmax([x['acc'] for x in log])
        print 'Maximum accuracy with this layer obtained at:',log[acc_max_index]
        if log[acc_max_index]['acc']>best_acc:
            best_acc = log[acc_max_index]['acc']
            best_arch = nn_architechture['n_neurons_in_hidden_layers'][l] = log[acc_max_index]['nn'][l]
        else:
            print 'Adding this layer did not improve accuracy. Terminating with the last best found network'
            break
        del a,ls,lg
    print '\n\n\nBest architecture is found to be: ',best_arch,', with an accuracy of: ',best_acc*100,'%'
    return best_arch
def plot_network_accuracies(log):
    accs = [x['acc'] for x in log]
    nns = [x['nn'] for x in log]
    acc_max_index = np.argmax([x['acc'] for x in log])
    text = 'Maximum accuracy at '+str(log[acc_max_index]['nn']) + ',  ' + '{0:.5f}'.format(log[acc_max_index]['acc'])
    y_pos = np.arange(len(nns))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y_pos, accs, linewidth=3)
    ax.annotate(text, xy=(acc_max_index, log[acc_max_index]['acc']), xytext=(len(nns)/2,log[acc_max_index]['acc']+0.01),
            arrowprops=dict(facecolor='black'),
            )
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Network architectures')
    plt.tight_layout(pad=3)
    plt.savefig(directories['plots']+'/comparison_best_nn.png')
    plt.pause(0.1)
    plt.close()
def plot_activation_accuracies(log):
    accs = [x['acc'] for x in log]
    acs = [x['activation'] for x in log]
    y_pos = np.arange(len(acs))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(y_pos, accs, align='center', alpha=0.5)
    plt.xticks(y_pos, acs)
    plt.ylabel('Accuracy')
    plt.savefig(directories['plots']+'/comparison_best_activation.png')
    plt.pause(0.1)
    plt.close()
def plot_dropout_accuracies(log):
    accs = [x['acc'] for x in log]
    acs = [x['dropout'] for x in log]
    y_pos = np.arange(len(acs))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(y_pos, accs, align='center', alpha=0.5)
    plt.xticks(y_pos, acs, rotation=45)
    plt.ylabel('Accuracy')
    plt.savefig(directories['plots']+'/comparison_best_dropout.png')
    plt.pause(0.1)
    plt.close()
def plot_cost_accuracies(log):
    accs = [x['acc'] for x in log]
    acs = [x['cost'] for x in log]
    y_pos = np.arange(len(acs))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(y_pos, accs, align='center', alpha=0.5)
    plt.xticks(y_pos, acs, rotation=45)
    plt.ylabel('Accuracy')
    plt.savefig(directories['plots']+'/comparison_best_cost.png')
    plt.pause(0.1)
    plt.close()
'''
