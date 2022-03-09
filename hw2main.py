import hw2
import hw2_utils
import torch
import matplotlib.pyplot as plt
import numpy as np

X, y = hw2_utils.xor_data()


alpha = hw2.svm_solver(X, y, 0.1, 10000, hw2_utils.poly(degree=2))

plots = np.linspace(-5, 5, 10)

predfxn = lambda x : hw2.svm_solver(x, y, 0.1, 10000, hw2_utils.poly(degree=2))

hw2_utils.svm_contour(predfxn, -5, 5, -5, 5)
