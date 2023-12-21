"""
A simple walkthrough of how to code a convolutional neural network (CNN)
using the PyTorch library. For demonstration we train it on the very
common MNIST dataset of handwritten digits. In this code we go through
how to create the network as well as initialize a loss function, optimizer,
check accuracy and more.

Programmed by Aladdin Persson
* 2020-04-08: Initial coding
* 2021-03-24: More detailed comments and small revision of the code
* 2022-12-19: Small revision of code, checked that it works with latest PyTorch version

"""

import numpy as np
import os

path = r'Results/'


import os
import glob

def write_to_file(filename):
    with open(os.path.join('', filename), 'w') as f:
        pass  # do nothing if the file already exists

# TL_Clf_result_S1_Test-6-30

for i in range(6, 10):
    for filename in glob.glob("Results/TL_Clf_result_S1_Test-{}-*.csv".format(i)):
        write_to_file(filename)

print("All data compiled and saved to result.csv")

# a = np.array([[x for x in range(22)]])
# b = np.array([[1 for x in range(22)]])
#
# c = np.append(a, b, axis=0)
# print(np.shape(c))
# diff = np.setdiff1d(a, b)
# app = np.append(a, b, axis=0).T
#
# delete = np.delete(a, [1, 4, 5])
#
# b2 = np.array([1, 8, 12, 16])
# b3 = np.array([2, 9, 13, 17])
# b_diff = np.append(b2, b3)
# b4 = np.setdiff1d(np.array([x for x in range(1, 22)]), b_diff)
#
# # def pad_0(array_2d, len=15):
# #     result = []
# #     for array_1d in array_2d:
# #         while len(array_2d) < 15:
#
#
#
#
# from numpy import genfromtxt
# my_data = genfromtxt('Results/TL_Clf_result_S2_Test-30-30.csv', delimiter=',')
#
# print(my_data)