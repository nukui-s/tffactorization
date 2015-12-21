"""Non-negative Matrix Factorization on TensorFlow

Authors : Shun Nukui
License : GNU General Public License v2.0

"""
from __future__ import division
import numpy as np
import tensorflow as tf

class TFNMF(object):
    """class for Non-negative Matrix Factorization on TensorFlow

    Requirements:
        TensorFlow version >= 0.6
    """
    def __init__(self, V, rank):

        #convert numpy matrix(2D-array) into TF Tensor
        V_ = tf.constant(V, dtype=tf.float32)
        shape = V.shape

        #scale uniform random with sqrt(V.mean() / rank)
        scale = 2 * np.sqrt(V.mean() / rank)
        initializer = tf.random_uniform_initializer(maxval=scale)

        graph = tf.get_default_graph()

        self.H = H = tf.get_variable("H", [rank, shape[1]],
                                     initializer=initializer)
        self.W = W = tf.get_variable(name="W", shape=[shape[0], rank],
                                     initializer=initializer)

        #save W for calculating delta with the updated W
        W_old = tf.get_variable(name="W_old", shape=[shape[0], rank])
        save_W = W_old.assign(W)

        #Multiplicative updates
        with graph.control_dependencies([save_W]):
            #update operation for H
            Wt = tf.transpose(W)
            WV = tf.matmul(Wt, V_)
            WWH = tf.matmul(tf.matmul(Wt, W), H)
            WV_WWH = WV / WWH
            #select op should be executed in CPU not in GPU
            with tf.device('/cpu:0'):
                #convert nan to zero
                WV_WWH = tf.select(tf.is_nan(WV_WWH),
                                    tf.zeros_like(WV_WWH),
                                    WV_WWH)
            H_new = H * WV_WWH
            update_H = H.assign(H_new)

        with graph.control_dependencies([save_W, update_H]):
            #update operation for W (after updating H)
            Ht = tf.transpose(H)
            VH = tf.matmul(V_, Ht)
            WHH = tf.matmul(W, tf.matmul(H, Ht))
            VH_WHH = VH / WHH
            with tf.device('/cpu:0'):
                VH_WHH = tf.select(tf.is_nan(VH_WHH),
                                        tf.zeros_like(VH_WHH),
                                        VH_WHH)
            W_new = W * VH_WHH
            update_W = W.assign(W_new)

        self.delta = tf.reduce_sum(tf.abs(W_old - W))

        self.step = tf.group(save_W, update_H, update_W)

    def run(self, sess, max_iter=200, min_delta=0.001):
        tf.initialize_all_variables().run()
        for i in range(max_iter):
            self.step.run()
            delta = self.delta.eval()
            if delta < min_delta:
                break
        W = self.W.eval()
        H = self.H.eval()
        return W, H
