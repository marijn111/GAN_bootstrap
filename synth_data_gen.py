import tensorflow as tf
from tf_slim import layers as _layers
tf.compat.v1.disable_eager_execution()
import numpy as np
from utils import extract_time, rnn_cell, random_generator
import argparse
import os
import pickle


class SynthDataGen:
    def __init__(self, args):
        self.args = args

    def main(self):
        tf.compat.v1.reset_default_graph()

        # Input placeholders
        self.X = tf.compat.v1.placeholder(tf.float32, [None, 24, 1], name="myinput_x")
        self.Z = tf.compat.v1.placeholder(tf.float32, [None, 24, 1], name="myinput_z")
        self.T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")

        # Generator
        self.E_hat = self.generator(self.Z, self.T)
        self.H_hat = self.supervisor(self.E_hat, self.T)
        self.X_hat = self.recovery(self.H_hat, self.T)

        saver = tf.compat.v1.train.Saver()

        final_output = {}

        self.ori_data, self.min_val, self.max_val, real_mean = self.data_loading()
        final_output['real_mean'] = real_mean

        # Basic Parameters
        self.no, self.seq_len, self.dim = np.asarray(self.ori_data).shape

        # Maximum sequence length and each sequence length
        self.ori_time, self.max_seq_len = extract_time(self.ori_data)

        self.z_dim = self.dim

        model_path = os.path.join(self.args.model_path, self.args.dgp_type, str(self.args.run_id))

        with tf.compat.v1.Session() as sess:
            saver.restore(sess, os.path.join(model_path, f'model_{self.args.run_id}.ckpt'))
            final_output['synth_means'] = []

            for j in range(self.args.synth_data_len):
                synth_data = self.generate_synthethic_data(sess)

                ori_data_renormalized = synth_data * np.array([self.max_val])
                ori_data_renormalized = ori_data_renormalized + np.array([self.min_val])

                generated_data = list()

                for it in range(417):
                    temp = ori_data_renormalized[it, :self.args.seq_len, :]
                    mean = np.mean(temp, axis=0)
                    generated_data.append(mean)
                repl_mean_first_x = np.mean(generated_data)
                final_output['synth_means'].append(repl_mean_first_x)

        return final_output

    def MinMaxScaler(self, data):
        """Min-Max Normalizer.

        Args:
        - data: raw data

        Returns:
        - norm_data: normalized data
        - min_val: minimum values (for renormalization)
        - max_val: maximum values (for renormalization)
        """
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    def data_loading(self):
        data_path = os.path.join(self.args.data_path, self.args.dgp_type, f'{self.args.dgp_type}_10k_{self.args.run_id}.csv')
        ori_data = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=1)
        real_mean = np.mean(ori_data)
        ori_data, min_val, max_val = self.MinMaxScaler(ori_data)
        # Preprocess the dataset
        temp_data = []
        # Cut data by sequence length
        for i in range(0, len(ori_data) - self.args.seq_len):
            _x = ori_data[i:i + self.args.seq_len]
            _x = np.vstack(_x)

            temp_data.append(_x)

        # Mix the datasets (to make it similar to i.i.d)
        idx = np.random.permutation(len(temp_data))
        data = []
        for i in range(len(temp_data)):
            data.append(temp_data[idx[i]])
        return data, min_val, max_val, real_mean

    def recovery(self, H, T):
        """Recovery network from latent space to original space.

        Args:
        - H: latent representation
        - T: input time information

        Returns:
        - X_tilde: recovered data
        """
        with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
            r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [rnn_cell(self.args.module, self.args.hidden_dim) for _ in range(self.args.num_layer)])
            r_outputs, final_state = tf.compat.v1.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length=T)
            X_tilde = _layers.fully_connected(r_outputs, 1, activation_fn=tf.nn.sigmoid)

        return X_tilde

    def generator(self, Z, T):
        """Generator function: Generate time-series data in latent space.

        Args:
        - Z: random variables
        - T: input time information

        Returns:
        - E: generated embedding
        """
        with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
            g_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [rnn_cell(self.args.module, self.args.hidden_dim) for _ in range(self.args.num_layer)])
            g_outputs, final_state = tf.compat.v1.nn.dynamic_rnn(g_cell, Z, dtype=tf.float32, sequence_length=T)
            E = _layers.fully_connected(g_outputs, self.args.hidden_dim, activation_fn=tf.nn.sigmoid)

        return E

    def supervisor(self, H, T):
        """Generate next sequence using the previous sequence.

        Args:
        - H: latent representation
        - T: input time information

        Returns:
        - S: generated sequence based on the latent representations generated by the generator
        """
        with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
            s_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [rnn_cell(self.args.module, self.args.hidden_dim) for _ in range(self.args.num_layer - 1)])
            s_outputs, final_state = tf.compat.v1.nn.dynamic_rnn(s_cell, H, dtype=tf.float32, sequence_length=T)
            S = _layers.fully_connected(s_outputs, self.args.hidden_dim, activation_fn=tf.nn.sigmoid)

        return S

    def generate_synthethic_data(self, sess):

        Z_mb = random_generator(self.no, self.z_dim, self.ori_time, self.max_seq_len)
        return sess.run(self.X_hat,
                        feed_dict={self.Z: Z_mb, self.X: self.ori_data, self.T: self.ori_time})

    def save_output(self, final_output):
        with open(os.path.join(self.args.output_path, self.args.dgp_type, f'final_output_run_{self.args.run_id}.pkl'), 'wb') as f:
            pickle.dump(final_output, f)


def main(args):
    synth_generator = SynthDataGen(args)
    final_output = synth_generator.main()
    synth_generator.save_output(final_output)


if __name__ == '__main__':

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seq_len',
        default=24,
        type=int
    )
    parser.add_argument(
        '--data_path',
        type=str,
    )
    parser.add_argument(
        '--module',
        choices=['gru'],
        default='gru',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=24,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=3,
        type=int)
    parser.add_argument(
        '--run_id',
        help='run id',
        type=int,
        default=1
    )
    parser.add_argument(
        '--output_path',
        help='path to the output',
        type=str,
    )
    parser.add_argument(
        '--dgp_type',
        help='type of dgp',
        choices=['arma', 'ar', 'ma', 'farima', 'garch'],
        type=str,
        default='arma'
    )
    parser.add_argument(
        '--synth_data_len',
        help='length of the synthetic data',
        type=int,
        default=600
    )
    parser.add_argument(
        '--model_path',
        help='path to the model',
        type=str,
    )
    args = parser.parse_args()
    main(args)
