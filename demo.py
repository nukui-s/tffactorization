#!/usr/bin/python
"""Demo script for running algorithm

Authors : Shun Nukui
License : GNU General Public License v2.0

Usage: python demo.py
"""
from __future__ import print_function
import time
import os
import numpy as np
import tensorflow as tf
from tfnmf import TFNMF


def main():
    #user setting parameters
    V = np.random.rand(3000,1000)
    rank = 2
    num_core = 8
    #available algorithms: mu, grad, mgpu
    algo = "mu"
    os.system("rm -rf tfnmflog")

    tfnmf = TFNMF(V, rank, algo=algo)

    config = tf.ConfigProto(inter_op_parallelism_threads=num_core,
                                       intra_op_parallelism_threads=num_core)
    #Multiplicative Update algorithm
    with tf.Session(config=config) as sess:
        start = time.time()
        W, H = tfnmf.run(sess)
        print("Computational Time for TFNMF: ", time.time() - start)

    W = np.mat(W)
    H = np.mat(H)
    error = np.power(V - W * H, 2).sum()
    print("Reconstruction Error for TFNMF: ", error)

if __name__ == '__main__':
     main()
