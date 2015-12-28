"""Non-negative Matrix Factorization on TensorFlow

Authors : Shun Nukui
License : GNU General Public License v2.0

"""
from __future__ import division
import numpy as np
import tensorflow as tf

INFINITY = 10e+9
EPSILON = 10e-9


class TFNMF(object):
    """class for Non-negative Matrix Factorization on TensorFlow

    Requirements:
        TensorFlow version >= 0.6
    """
    def __init__(self, V, rank, algo="mu", learning_rate=0.01, constraint=10e+1):

        #convert numpy matrix(2D-array) into TF Tensor
        self.V = tf.constant(V, dtype=tf.float32)
        shape = V.shape

        self.rank = rank
        self.algo = algo
        #used for AdagradOptimiber
        self.lr = learning_rate
        self.constraint = constraint

        #scale uniform random with sqrt(V.mean() / rank)
        scale = 2 * np.sqrt(V.mean() / rank)
        initializer = tf.random_uniform_initializer(minval=0.0, maxval=scale)

        self.H =  tf.get_variable("H", [rank, shape[1]],
                                     initializer=initializer)
        self.W =  tf.get_variable(name="W", shape=[shape[0], rank],
                                     initializer=initializer)

        if algo == "mu":
            self._build_mu_algorithm()
        elif algo == "grad":
            self._build_grad_algorithm()
        elif algo == "mgpu":
            self._build_mGPU_algorithm()
        else:
            raise ValueError("The attribute algo must be in {'mu', 'grad', 'mgpu'}")

    def _build_mGPU_algorithm(self):
        """build dataflow graph for NMF-mGPU algorithm

            References
            --------------
            E. Mejia-Roa, D. Tabas-Madrid, J. Setoain, C. Garcia, F. Tirado and
            A. Pascual-Montano. NMF-mGPU: Non-negative matrix factorization
            on multi-GPU systems. BMC Bioinformatics 2015,
            16:43. doi:10.1186/s12859-015-0485-4
            [http://www.biomedcentral.com/1471-2105/16/43]
        """

        graph = tf.get_default_graph()
        V, H, W = self.V, self.H, self.W
        #avoid inf caused by zero division
        WH_inv = 1 / (tf.matmul(W, H) + EPSILON)
        U = V * WH_inv

        WtU = tf.matmul(tf.transpose(W), U)
        W_inv = 1 / tf.reduce_sum(W, reduction_indices=0)
        W_inv = tf.diag(W_inv)
        H_new = H * tf.matmul(W_inv, WtU)
        update_H = tf.assign(H, H_new)

        with graph.control_dependencies([update_H]):
            UHt = tf.matmul(U, tf.transpose(H))
            H_inv = 1 / tf.reduce_sum(H, reduction_indices=1)
            H_inv = tf.diag(H_inv)
            W_new = W * tf.matmul(UHt, H_inv)
            update_W = tf.assign(W, W_new)

        self.optimize = tf.group(update_H, update_W)

        self.loss = tf.reduce_sum(tf.pow(V - tf.matmul(W, H), 2))
        loss_summ = tf.scalar_summary("loss_mgpu", self.loss)
        self.merged = tf.merge_all_summaries()


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
        nn_constraint = self.constraint * (nn_w + nn_h)

        self.loss = loss = f_norm + nn_constraint
        self.optimize = tf.train.AdagradOptimizer(self.lr).minimize(loss)

        loss_summ = tf.scalar_summary("loss_grad", loss)
        self.merged = tf.merge_all_summaries()


    def _build_mu_algorithm(self):
        """build dataflow graph for Multiplicative update algorithm"""

        V, H, W = self.V, self.H, self.W
        rank = self.rank
        shape = V.get_shape()

        graph = tf.get_default_graph()

        #Multiplicative updates
        #update operation for H
        Wt = tf.transpose(W)
        WV = tf.matmul(Wt, V)
        WWH = tf.matmul(tf.matmul(Wt, W), H)
        WV_WWH = WV / (WWH + EPSILON)
        H_new = H * WV_WWH
        update_H = H.assign(H_new)

        with graph.control_dependencies([update_H]):
            #update operation for W (after updating H)
            Ht = tf.transpose(H)
            VH = tf.matmul(V, Ht)
            WHH = tf.matmul(W, tf.matmul(H, Ht))
            VH_WHH = VH / (WHH + EPSILON)
            W_new = W * VH_WHH
            update_W = W.assign(W_new)

        self.optimize = tf.group(update_H, update_W)

        self.loss = tf.reduce_sum(tf.pow(V - tf.matmul(W, H), 2))
        loss_summ = tf.scalar_summary("loss_mu", self.loss)
        self.merged = tf.merge_all_summaries()


    def run(self, sess, max_iter=500, min_delta=0.001, logfile="tfnmflog"):
        writer = tf.train.SummaryWriter(logfile, sess.graph_def)
        tf.initialize_all_variables().run()
        pre_loss = INFINITY
        for i in range(max_iter):
            loss, merged, _ = sess.run([self.loss, self.merged,
                                                        self.optimize])
            if pre_loss - loss < min_delta:
                break
            writer.add_summary(merged, i)
            pre_loss = loss
        W = self.W.eval()
        H = self.H.eval()
        return W, H
