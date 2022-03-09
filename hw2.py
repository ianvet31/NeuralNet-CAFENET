import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    n = x_train.shape[0]

    alpha = torch.zeros((n, 1), requires_grad=True)

    for i in range(num_iters):
        G = torch.zeros(n, n)
        for x, xi in enumerate(x_train):
            for y, xj in enumerate(x_train):
                G[x, y] = y_train[x] * y_train[y] * kernel(xi, xj)


        #f = 1/2 * (torch.sum(torch.mul(alpha, alpha) * torch.mul(y_train, y_train) * kernel(x_train, x_train)) - torch.sum(alpha)
        
        f = (torch.sum(alpha)) - (1/2 * ((alpha.T @ (G) @ alpha)))
        func = -f
        func.backward()

        with torch.no_grad():
            a = alpha - (lr * alpha.grad)
            if c != None:
                alpha = a.clamp_(0, c)
            else:
                alpha = a.clamp_(0)

            alpha.requires_grad_()

    return alpha
    


def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''

    m = x_test.shape[0]
    n = x_train.shape[0]
    y_test = torch.zeros(m, 1)

    for i in range(m):
        for j in range(n):
            y_test[i] += alpha[j] * y_train[j] * kernel(x_train[j], x_test[i])
    
    return y_test

class CAFENet(nn.Module):
    def __init__(self):
        '''
            Initialize the CAFENet by calling the superclass' constructor
            and initializing a linear layer to use in forward().

            Arguments:
                self: This object.
        '''
        super(CAFENet, self).__init__()
        self.layer = nn.Linear(380*240, 6)



    def forward(self, x):
        '''
            Computes the network's forward pass on the input tensor.
            Does not apply a softmax or other activation functions.

            Arguments:
                self: This object.
                x: The tensor to compute the forward pass on.
        '''
        x = self.layer(x)
        return x
        

def fit(net, X, y, n_epochs=201):
   
    optimizer = torch.optim.Adam(net.parameters())
    losses = []
    for i in range(n_epochs):
        net.zero_grad()
        yhat = net(X)
        loss = torch.nn.CrossEntropyLoss()(yhat, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            losses.append(loss.detach())

    return losses




def plot_cafe_loss():
    '''
    Trains a CAFENet on the CAFE dataset and plots the zero'th through 200'th
    epoch's losses after training. Saves the trained network for use in
    visualize_weights.
    '''
    X, y = hw2_utils.get_cafe_data()
    net = CAFENet()
    losses = fit(net, X, y)
    epochs = np.linspace(0, 201, 201)

    torch.save(net, "net.pt")

    plt.plot(epochs, losses)
    plt.show()

def visualize_weights():
    '''
    Loads the CAFENet trained in plot_cafe_loss, maps the weights to the grayscale
    range, reshapes the weights into the original CAFE image dimensions, and
    plots the weights, displaying the six weight tensors corresponding to the six
    labels.
    '''
    net = torch.load("net.pt")
    weights = net.layer.parameters()
    maxweight = 0
    minweight = 1
    for i in weights:
        for j in range(6):
            print(j)
            w = i[j]
            if w.max() > maxweight:
                maxweight = w.max()

    print(maxweight)

def print_confusion_matrix():
    '''
    Loads the CAFENet trained in plot_cafe_loss, loads training and testing data
    from the CAFE dataset, computes the confusion matrices for both the
    training and testing data, and prints them out.
    '''

    net = torch.load("net.pt")
    X, y = hw2_utils.get_cafe_data()
    X_test, y_test = hw2_utils.get_cafe_data("test")

    yhats = net.forward(X)
    yhats_test = net.forward(X_test)
    ybutts = torch.argmax(yhats, 1)
    ybutts_test = torch.argmax(yhats_test, 1)

    print(confusion_matrix(y, ybutts))
    print(confusion_matrix(y_test, ybutts_test))


