import numpy as np
import matplotlib.pyplot as plt
from activation import *

def initialize_parameters(layer_dims):
    
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
    
        parameters['W'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))
        
    return parameters

def linear_forward(A, W, b):
    Z=np.dot(W,A)+b
    
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        
        Z, linear_cache=linear_forward(A_prev, W, b)
        A, activation_cache=sigmoid(Z)
    
    elif activation == "relu":
        
        Z, linear_cache=linear_forward(A_prev, W, b)
        A, activation_cache=relu(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2    # number of layers in the neural network

    for l in range(1, L):
        A_prev = A 

        A, cache=linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation="relu")
        caches.append(cache)

    AL, cache=linear_activation_forward(A,parameters['W'+str(L)], parameters['b'+str(L)], activation="sigmoid")
    caches.append(cache)
          
    return AL, caches

def compute_cost(AL, Y):
 
    m = Y.shape[1]

    cost= np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)),axis=1,keepdims=True)/-m
    
    cost = np.squeeze(cost)   
    
    return cost

def linear_backward(dZ, cache):
  
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
  
    linear_cache, activation_cache = cache
    
    if activation == "relu":
    
        dZ=relu_backward(dA,activation_cache)
        dA_prev, dW, db=linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
      
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db=linear_backward(dZ, linear_cache)        
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
   
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
  
    dAL=-(np.divide(Y,AL)-np.divide((1-Y),(1-AL)))
    
    current_cache=caches[L-1]
    dA_prev_temp, dW_temp, db_temp =linear_activation_backward(dAL, current_cache, activation="sigmoid")
    grads["dA" + str(L-1)] =dA_prev_temp
    grads["dW" + str(L)] =dW_temp
    grads["db" + str(L)] =db_temp
  
    for l in reversed(range(L-1)):
      
        current_cache =caches[l]
        dA_prev_temp, dW_temp, db_temp =linear_activation_backward(grads["dA"+str(l+1)], current_cache, activation="relu")
        grads["dA" + str(l)] =dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(params, grads, learning_rate):
   
    parameters = params.copy()
    L = len(parameters) // 2 

    for l in range(L):
       
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)] 
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)] 

    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.025, num_iterations = 3000, print_cost=False):
    
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters(layers_dims)
    
    for i in range(0, num_iterations):

        # Forward propagation:
        AL, caches = L_model_forward(X, parameters)
        
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
       
        grads = L_model_backward(AL, Y, caches)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs

def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    #print(probas.shape)
    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

def print_mislabeled_images(classes, X, y, p):
    
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
        
def plot_costs(costs, learning_rate=0.025):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()