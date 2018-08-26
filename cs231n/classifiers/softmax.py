import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_classes = W.shape[1]
  num_train = X.shape[0]

  
  for i in range(num_train):
    scores = X[i].dot(W)
    #отнимаем максимальные очки, так, что везде остаются лишь нули или отрицательные числа
    #каждый элемент делим на сумму всех десяти элементов
    softmax = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores-np.max(scores)),keepdims = True)
    for j in range(num_classes):
      if j == y[i]:
        loss -= np.log(softmax[j])
        dW[:,j]+=(softmax[j]-1)*X[i]
      else:
        dW[:,j]+=softmax[j]*X[i]
      
      
      
    
    
  loss /= num_train  
  loss+=  reg * np.sum(W * W)
  dW = dW / num_train
  dW = dW + reg*W  
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #нормализую весы, ради численной стабильности
  sc1 = scores - (np.max(scores,axis = 1)).reshape(num_train,1)
  #вот она самая, функция софтмакс
  softmax = np.exp(sc1) / np.sum(np.exp(sc1),axis=1).reshape(num_train,1)
  loss = np.sum(- sc1[range(num_train), y] + np.log(np.sum(np.exp(sc1),axis=1))) 
  loss /= num_train  
  loss+= 0.5* reg * np.sum(W * W) 
  softmax[range(num_train), y] -=1  
  dW = X.T.dot(softmax) 
  dW /= num_train
  dW +=  reg*W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

