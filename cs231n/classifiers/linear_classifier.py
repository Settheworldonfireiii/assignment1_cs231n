from __future__ import print_function

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    if self.W is None:
      # lazily initialize W
      self.W = 0.001 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for i in range(num_iters):
      
      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      indicies = np.random.choice(range(num_train),batch_size,replace = True)
      X_batch = np.array(X[indicies])
      y_batch = np.array(y[indicies])
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)
      #learning rate decay
      lr_decay = 0.998
      #learning_rate = learning_rate*lr_decay
      #print(learning_rate)
      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      self.W -= grad*learning_rate
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and i % 100 == 0:
        print('iteration %d / %d: loss %f' % (i, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.
    
     Inputs:
     - X: A numpy array of shape (N, D) containing training data; there are N
       training samples each of dimension D.

     Returns:
     - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    scores= X.dot(self.W)        
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
   
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
      """
      Structured SVM loss function, vectorized implementation.

      Inputs and outputs are the same as svm_loss_naive.
      """
      loss = 0.0
      dW = np.zeros(W.shape) # initialize the gradient as zero
      num_classes = W.shape[1]
      num_train = X.shape[0]

      #############################################################################
      # TODO:                                                                     #
      # Implement a vectorized version of the structured SVM loss, storing the    #
      # result in loss.                                                           #
      #############################################################################
      scores = X.dot(W)
      x = np.arange(num_train)
      #фильтруем, выбирая лишь нужные классы  
      correct_class_score = scores[x, y]
      #отнимаю от всех очков правильные очки и прибавляя 1(можно взять другое число)
      #получая меру того, насколько больше очки некоего неправильно классифицированного
      #класса, чем правильного.
      margins = np.maximum(scores - correct_class_score.reshape(num_train, 1) + 1.0, 0)
      #зануляю правильные очки, чтобы не плюсовать их к мере неправильности нынешних весов
      margins[x, y] = 0
  
      loss = np.sum(margins)/ num_train

      # Add regularization to the loss.
      loss +=0.5* reg * np.sum(W * W)

  
      mrgns1 = np.zeros(margins.shape)
      mrgns1[margins>0] = 1
      mrgns1[x, y] -= np.sum(mrgns1, axis=1)
      dW = X.T.dot(mrgns1)
      dW /= num_train
      dW += reg * W 
      return loss, dW


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

