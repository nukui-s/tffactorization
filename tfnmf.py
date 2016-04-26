"""Non-negative Matrix Factorization on TensorFlow

Authors : Shun Nukui
License : GNU General Public License v2.0

"""
from __future__ import division
import numpy as np
import tensorflow as tf

INFINITY = 10e+12

class TFNMF(object):
    """class for Non-negative Matrix Factorization on TensorFlow

    Requirements:
        TensorFlow version >= 0.6
    """
    def __init__(self, V, rank, algo="mu", learning_rate=0.01):

        #convert numpy matrix(2D-array) into TF Tensor
        self.V = tf.constant(V, dtype=tf.float32)
        shape = V.shape

        self.rank = rank
        self.algo = algo
        self.lr = learning_rate

        #scale uniform random with sqrt(V.mean() / rank)
        scale = 2 * np.sqrt(V.mean() / rank)
        initializer = tf.random_uniform_initializer(maxval=scale)

        self.H =  tf.get_variable("H", [rank, shape[1]],
                                     initializer=initializer)
        self.W =  tf.get_variable(name="W", shape=[shape[0], rank],
                                     initializer=initializer)

        if algo == "mu":
            self._build_mu_algorithm()
        elif algo == "grad":
            self._build_grad_algorithm()
        else:
            raise ValueError("The attribute algo must be in {'mu', 'grad'}")

    def _build_grad_algorithm(self):
        """build dataflow graph for optimization with Adagrad algorithm"""

        V, H, W = self.V, self.H, self.W

        WH = tf.matmul(W, H)

        #cost of Frobenius norm
        f_norm = tf.reduce_sum(tf.pow(V - WH, 2))

        #Non-negative constraint
        #If all elements positive, constraint will be 0
        nn_w = tf.reduce_sum(tf.abs(W) - W)
        nn_h = tf.reduce_sum(tf.abs(H) - H)
        constraint = INFINITY * (nn_w + nn_h)

        self.loss = loss= f_norm + constraint
        self.optimize = tf.train.AdagradOptimizer(self.lr).minimize(loss)


    def _build_mu_algorithm(self):
        """build dataflow graph for Multiplicative algorithm"""

        V, H, W = self.V, self.H, self.W
        rank = self.rank
        shape = V.get_shape()

        graph = tf.get_default_graph()

        #save W for calculating delta with the updated W
        W_old = tf.get_variable(name="W_old", shape=[shape[0], rank])
        save_W = W_old.assign(W)

        #Multiplicative updates
        with graph.control_dependencies([save_W]):
            #update operation for H
            Wt = tf.transpose(W)
            WV = tf.matmul(Wt, V)
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
            VH = tf.matmul(V, Ht)
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
        algo = self.algo

        tf.initialize_all_variables().run()

        if algo == "mu":
            return self._run_mu(sess, max_iter, min_delta)
        elif algo == "grad":
            return self._run_grad(sess, max_iter, min_delta)
        else:
            raise ValueError

    def _run_mu(self, sess, max_iter, min_delta):
        for i in xrange(max_iter):
            self.step.run()
            delta = self.delta.eval()
            if delta < min_delta:
                break
        W = self.W.eval()
        H = self.H.eval()
        return W, H

    def _run_grad(self, sess, max_iter, min_delta):
        pre_loss = INFINITY
        for i in xrange(max_iter):
            loss, _ = sess.run([self.loss, self.optimize])
            if pre_loss - loss < min_delta:
                break
            pre_loss = loss
        W = self.W.eval()
        H = self.H.eval()
        return W, H
