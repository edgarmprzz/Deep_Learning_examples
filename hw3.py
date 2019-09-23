import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import time
import subprocess
import glob

# Parameters for finding best network architecture
max_hidden_conv_layers = 3
n_kernel_per_conv_layer = [16,64,256]
n_size_kernel = [1,3,5,7]
max_hidden_fc_layers = 2
n_neurons_per_layer = [128,256,1024]
kernel_shapes = ['NxN','1xN','Nx1','Mx1']
activations = ['relu','sigmoid','tanh']
costs = ['softmax_cross_entropy_logits','sigmoid_cross_entropy_logits','hinge_loss']
learning_rates = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
dropouts = [[1.0,1.0,1.0,1.0],[0.7,1.0,1.0,1.0],[1.0,0.7,1.0,1.0],[0.7,0.7,1.0,1.0],[0.7,1.0,1.0,0.5],[1.0,0.7,1.0,0.5],[0.7,0.7,1.0,0.5],[1.0,1.0,1.0,0.5]]

directories = {
    'save_weights':'./saved_weights',
    'tensorboard_graph':'./graphs',
    'plots':'./plots'
}

def check_create_directories(directories):
    for directory in directories:
        if not os.path.exists(directories[directory]):
            os.makedirs(directories[directory])

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

def plot_accuracies(log,variable):
    accs = [x['acc'] for x in log]
    acc_max_index = np.argmax(accs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if variable == 'network':
        var = [x['nn']['n_size_and_kernel_in_hidden_layers']+x['nn']['n_fc_neurons_in_hidden_layers'] for x in log]
        y_pos = np.arange(len(var))
        text = 'Maximum accuracy at '+str(log[acc_max_index]['nn']) + ',  ' + '{0:.5f}'.format(log[acc_max_index]['acc'])
        ax.plot(y_pos, accs, linewidth=3)
        ax.annotate(text, xy=(acc_max_index, log[acc_max_index]['acc']), xytext=(len(var)/2,log[acc_max_index]['acc']+0.01),
                arrowprops=dict(facecolor='black'),
                )
        ax.set_xlabel('Network architectures')
        plt.tight_layout(pad=3)
    else:
        var = None
        if variable == 'kernel':
            var = [kernel_shapes[x['nn'][variable]] for x in log]
        else:
            var = [x['nn'][variable] for x in log]
        y_pos = np.arange(len(var))
        plt.tight_layout(pad=5)
        plt.bar(y_pos, accs, align='center', alpha=0.5)
        plt.xticks(y_pos, var, rotation=20)
    ax.set_ylabel('Accuracy')
    plt.savefig(directories['plots']+'/comparison_best_'+variable+'.png')
    plt.pause(0.1)
    plt.close()

def find_best_network_from_all_networks():
    ## Finding the possible combinations of networks (CNN+FC)
    ## Assumption: number of features/neurons in subsequent layer is less than previous layer
    kernels = []
    for i in n_kernel_per_conv_layer:
        for j in n_size_kernel:
            kernels.append([j,i])
    cnn_combinations = []
    for l in range(max_hidden_conv_layers):
        for nn in itertools.product(kernels,repeat=l+1):
            if len(nn)>1:
                nn = sorted(nn, key=lambda n:n[1],reverse=True)
            nn = tuple([tuple(x) for x in nn])
            cnn_combinations.append(nn)
    cnn_combinations = sorted(set(cnn_combinations), key=lambda n:len(n))
    print 'Number of CNN combinations: ',len(cnn_combinations)
    fc_combinations = []
    for l in range(max_hidden_fc_layers):
        for nn in itertools.product(n_neurons_per_layer,repeat=l+1):
            if len(nn)>1:
                nn = sorted(nn, key=lambda n:n,reverse=True)
                nn = tuple(nn)
            fc_combinations.append(nn)
    fc_combinations = sorted(set(fc_combinations), key=lambda n:len(n))
    print 'Number of FC combinations: ',len(fc_combinations)
    nn_combinations = []
    for i in cnn_combinations:
        for j in fc_combinations:
            nn_combinations.append(i+j)
    print 'Totoal network combinations: ',len(nn_combinations)
    ## Training each network and logging the accuracies
    print '\n\nTraining on all network combinations...\n\n\n'
    for i,n in enumerate(nn_combinations):
        print '\n\nAnalysing network ',n
        s = 'python,training_processes.py,'
        n_cnn,n_fc = 0,0
        for layer in n:
            if type(layer) is type(()): # Its a CNN
                n_cnn += 1
                s += '--CNN'+str(n_cnn)+'='+str(layer[0])+','+str(layer[1])+','
            else: #Its a FC
                n_fc += 1
                s += '--FC'+str(n_fc)+'='+str(layer)+','
        s += '--ACTIVATION=relu,--PLOT,--ID='+str(i)
        subprocess.call(s.replace('=',',').split(','))
        time.sleep(5)
    # Log and plot accuracies obtained from all networks
    log = []
    for f in sorted(glob.glob(directories['plots']+'/log_*.npy')):
        log.append(np.reshape(np.load(f),(-1))[0])
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy with this layer obtained at:',log[acc_max_index]
    np.save(directories['plots']+'/log_nn',log)
    plot_accuracies(log,'network')
    return log[acc_max_index]['nn']['n_size_and_kernel_in_hidden_layers']+log[acc_max_index]['nn']['n_fc_neurons_in_hidden_layers']

def find_best_kernel_shape():
    for i in range(len(kernel_shapes)):
        print '\n\nAnalysing network with kernel shape ',kernel_shapes[i]
        s = 'python,training_processes.py,--CNN1=5,32,--CNN2=5,64,--FC1=1024,--PLOT,--GRAPH,--STEPS,'
        s += '--ID='+str(i)+','
        s += '--KERNEL='+str(i)
        subprocess.call(s.replace('=',',').split(','))
        time.sleep(5)
    log = []
    for f in range(len(kernel_shapes)):
        f = directories['plots']+'/log_'+str(f)+'.npy'
        log.append(np.reshape(np.load(f),(-1))[0])
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy was obtained with kernel of shape: ',kernel_shapes[acc_max_index]
    np.save(directories['plots']+'/log_kernel',log)
    plot_accuracies(log,'kernel')
    return log[acc_max_index]['nn']['kernel']

def find_best_activation_function():
    for i,activation in enumerate(activations):
        print '\n\nAnalysing network with activation ',activation
        s = 'python,training_processes.py,--CNN1=7,256,--CNN2=3,256,--CNN3=1,256,--FC1=1024,--PLOT,--GRAPH,--STEPS,'
        s += '--ID='+str(i)+','
        s += '--ACTIVATION='+activation
        subprocess.call(s.replace('=',',').split(','))
        time.sleep(5)
    log = []
    for f in range(len(activations)):
        f = directories['plots']+'/log_'+str(f)+'.npy'
        log.append(np.reshape(np.load(f),(-1))[0])
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy obtained with activation function:',activations[acc_max_index]
    np.save(directories['plots']+'/log_activation',log)
    plot_accuracies(log,'activation')
    return log[acc_max_index]['nn']['activation']

def find_best_dropouts():
    for i,d in enumerate(dropouts):
        print '\n\nAnalysing network with CNN dropout ',d[0:2],' FC dropout ',d[2]
        s = 'python,training_processes.py,--CNN1=7,256,--CNN2=3,256,--CNN3=1,256,--FC1=1024,--PLOT,--GRAPH,--STEPS,'
        s += '--ID='+str(i)+','
        s += '--DROPOUT_CNN='+str(d[0])+','+str(d[1])+','+str(d[2])+','
        s += '--DROPOUT_FC='+str(d[3])
        print s.replace('=',',').split(',')
        subprocess.call(s.replace('=',',').split(','))
        time.sleep(5)
    log = []
    for i in range(len(dropouts)):
        f = directories['plots']+'/log_'+str(i)+'.npy'
        log_data = np.reshape(np.load(f),(-1))[0]
        log_data['nn']['dropout'] = dropouts[i]
        log.append(log_data)
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy obtained with dropouts as :',log[acc_max_index]['nn']['dropout']
    np.save(directories['plots']+'/log_dropout',log)
    plot_accuracies(log,'dropout')
    return log[acc_max_index]['nn']['dropout']

def find_best_cost_function():
    for i,cost in enumerate(costs):
        print '\n\nAnalysing network with cost function as ',cost
        s = 'python,training_processes.py,--CNN1=7,256,--CNN2=3,256,--CNN3=1,256,--FC1=1024,--PLOT,--GRAPH,--STEPS,'
        s += '--ID='+str(i)+','
        s += '--COST='+cost
        subprocess.call(s.replace('=',',').split(','))
        time.sleep(5)
    log = []
    for f in range(len(costs)):
        f = directories['plots']+'/log_'+str(f)+'.npy'
        log.append(np.reshape(np.load(f),(-1))[0])
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy obtained with cost function:',costs[acc_max_index]
    np.save(directories['plots']+'/log_cost',log)
    plot_accuracies(log,'cost')
    return log[acc_max_index]['nn']['cost']

def find_best_learning_rate():
    for i,alpha in enumerate(learning_rates):
        print '\n\nAnalysing network with learning rate of ',alpha
        s = 'python,training_processes.py,--CNN1=7,256,--CNN2=3,256,--CNN3=1,256,--FC1=1024,--PLOT,--GRAPH,--STEPS,'
        s += '--ID='+str(i)+','
        s += '--COST=sigmoid_cross_entropy_logits,'
        s += '--ALPHA='+str(alpha)
        subprocess.call(s.replace('=',',').split(','))
        time.sleep(5)
    log,logs = [],[]
    for f in range(len(learning_rates)):
        f = directories['plots']+'/log_'+str(f)+'.npy'
        l = np.reshape(np.load(f),(-1))[0]
        log.append(l)
        logs.append([l['lg'],l['nn']['alpha']])
    acc_max_index = np.argmax([x['acc'] for x in log])
    print 'Maximum accuracy obtained with learning rate: ',learning_rates[acc_max_index]
    np.save(directories['plots']+'/log_learning_rate',log)
    np.save(directories['plots']+'/logs_alpha',logs)
    plot_accuracies(log,'alpha')
    plot_multi_learning(logs)
    return log[acc_max_index]['nn']['alpha']

def run():
    check_create_directories(directories)
    #### Q1 ####
    #subprocess.call('python,training_processes.py,--CNN1=5,32,--CNN2=5,64,--FC1=1024,--ACTIVATION=relu,--PLOT,--GRAPH,--SAVE,--STEPS'.replace('=',',').split(','))
    #### Q2 ####
    #best_network = find_best_network_from_all_networks()
    #### Q3 ####
    #best_kernel = find_best_kernel_shape()
    #### Q4 ####
    #best_activation = find_best_activation_function()
    #### Q5 ####
    best_dropout = find_best_dropouts()
    #### Q6 ####
    #best_cost = find_best_cost_function()
    #best_alpha = find_best_learning_rate()

if __name__ == "__main__":
    run()
