import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# color map
c_map = ['#2CBDFE', '#47DBCD', '#F3A0F2', 'yellow', '#661D98', '#F5B14C', 'm', 'k']

class Helpers:
    '''Helper functions for the TDQN agent.'''

    def __init__(self, batchsize, num_a, seq_len):
        self.batchsize = batchsize
        self.num_a = num_a
        self.seq_len = seq_len
        self.writer = None
        
    def tensorboard_writer(self, logdir):
        '''Instantiates a writer object.'''
        self.writer = tf.summary.FileWriter(logdir)
        
    def tensorboard_scalar(self, tag, value, step):
        '''Writes scalar values to tensorboard'''
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value), ])
        self.writer.add_summary(summary, step)
        
    def tensorboard_hist(self, tag, values, step, bins=1000):
        '''Computes a histogram given input values and writes to tensorboard.'''
        values = np.array(values)
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins)
        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))
        
        bin_edges = bin_edges[1:]
        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)
        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)

    def position_sizer(self, a):
        '''Maps an action preference to a position sizing parameter.

        Arguments:
        a: int [0, num_a], the argmax from the network output.

        Returns:
        sizer: int [1, num_a // 2], multipliers.
        Position: bool, True for long and False for short.

        '''
        # Measure up the number of position sizers
        num_sizers = self.num_a // 2
        # Polulate a list 
        sizers = []
        for _ in range(2):
            for i in range(num_sizers):
                sizers.append((i + 1) / num_sizers)
        # Select current sizer
        sizer = sizers[a]
        # Conclude long or short position
        if a < num_sizers:
            return sizer, True
        return sizer, False
    
    def unpack(self, transitions):
        '''Gets a batch from memory and slices it.'''
        S = [transitions[i][0] for i in range(self.batchsize)]
        A = np.array([transitions[i][1] for i in range(self.batchsize)])
        R = np.array([transitions[i][2] for i in range(self.batchsize)])
        S2 = [transitions[i][3] for i in range(self.batchsize)]
        T = np.array([transitions[i][4] for i in range(self.batchsize)])
        # stack respective input in S and S2
        x1_s, x1_s2 = S[0][0], S2[0][0]
        x2a_s, x2a_s2 = S[0][1], S2[0][1]
        x2b_s, x2b_s2 = S[0][2], S2[0][2]
        for i, j in zip(S[1:], S2[1:]):
            x1_s, x1_s2 = np.concatenate((x1_s, i[0])), np.concatenate((x1_s2, j[0]))
            x2a_s, x2a_s2 = np.concatenate((x2a_s, i[1])), np.concatenate((x2a_s2, j[1]))
            x2b_s, x2b_s2 = np.concatenate((x2b_s, i[2])), np.concatenate((x2b_s2, j[2]))
        S, S2 = [x1_s, x2a_s, x2b_s], [x1_s2, x2a_s2, x2b_s2]
        return S, A, R, S2, T
        
    def plot_train(self, episode, dists, P_value, Actions, filename, run):
        print(f'episode:', episode)
        sns.set_style('darkgrid')
        fig = plt.figure(figsize=(13, 6))
        # Cash etc
        plt.subplot(4, 1, 1)
        plt.plot(dists['cash'], c_map[0], label='cash')
        plt.plot(dists['stock_v'], c_map[1], label='stock_v')
        plt.legend(loc=('best'), frameon=False, ncol=2)
        plt.xticks([])
        plt.subplot(4, 1, 2)
        plt.plot(dists['stock_n'], c_map[3], label='stock_n')
        plt.legend(loc=('best'), frameon=False, ncol=1)
        plt.xticks([])
        # Portfolio return
        plt.subplot(4, 1, 3)
        plt.plot(P_value, c_map[2], label='portfolio value')
        plt.legend(loc=('best'), frameon=False, ncol=1)
        plt.xticks([])
        # actions
        plt.subplot(4, 1, 4)
        plt.bar(np.arange(self.num_a), height=list(Actions.values()), color=c_map[4])
        plt.xticks(np.arange(self.num_a))
        dir_path = 'outputs/' + filename + str(run)
        file = '{}/train_' + str(episode) + '.png'
        self.mkdir_p(dir_path)
        plt.savefig(file.format(dir_path))
        plt.show()

    def mkdir_p(self, mypath):
        '''Creates a directory. equivalent to using mkdir -p on the command line'''

        from errno import EEXIST
        from os import makedirs,path

        try:
            makedirs(mypath)
        except OSError as exc: # Python >2.5
            if exc.errno == EEXIST and path.isdir(mypath):
                pass
            else: raise