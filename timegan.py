"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

# Necessary Packages
import tensorflow as tf
from tf_slim.layers import layers as _layers

tf.compat.v1.disable_eager_execution()
from tqdm import tqdm, trange
from datetime import datetime
import os

import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator


class TimeGAN:
    def __init__(self, ori_data, parameters, min_value, max_value):
        self.ori_data = ori_data
        self.parameters = parameters
        self.min_val = min_value
        self.max_val = max_value
        print('Initiated TimeGAN.')



    def train(self):

        start_time_total = datetime.now()

        """TimeGAN function.

    Use original data as training set to generater synthetic data (time-series)

    Args:
      - ori_data: original time-series data
      - parameters: TimeGAN network parameters

    Returns:
      - generated_data: generated time-series data
    """
        # Initialization on the Graph
        tf.compat.v1.reset_default_graph()

        # Basic Parameters
        self.no, self.seq_len, self.dim = np.asarray(self.ori_data).shape

        # Maximum sequence length and each sequence length
        self.ori_time, self.max_seq_len = extract_time(self.ori_data)

        ## Build a RNN networks

        # Network Parameters
        hidden_dim = self.parameters['hidden_dim']
        num_layers = self.parameters['num_layer']
        iterations_embedding = self.parameters['iterations_embedding']
        iterations_supervised = self.parameters['iterations_supervised']
        iterations_joint = self.parameters['iterations_joint']
        batch_size = self.parameters['batch_size']
        module_name = self.parameters['module']
        self.z_dim = self.dim
        gamma = 1

        # Input placeholders
        self.X = tf.compat.v1.placeholder(tf.float32, [None, self.max_seq_len, self.dim], name="myinput_x")
        self.Z = tf.compat.v1.placeholder(tf.float32, [None, self.max_seq_len, self.z_dim], name="myinput_z")
        self.T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")

        def embedder(X, T):
            """Embedding network between original feature space to latent space.

      Args:
        - X: input time-series features
        - T: input time information

      Returns:
        - H: embeddings
      """
            with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
                e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                    [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
                e_outputs, final_state = tf.compat.v1.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)
                H = _layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)

            return H

        def recovery(H, T):
            """Recovery network from latent space to original space.

      Args:
        - H: latent representation
        - T: input time information

      Returns:
        - X_tilde: recovered data
      """
            with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
                r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                    [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
                r_outputs, final_state = tf.compat.v1.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length=T)
                X_tilde = _layers.fully_connected(r_outputs, self.dim, activation_fn=tf.nn.sigmoid)

            return X_tilde

        def generator(Z, T):
            """Generator function: Generate time-series data in latent space.

      Args:
        - Z: random variables
        - T: input time information

      Returns:
        - E: generated embedding
      """
            with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
                g_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                    [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
                g_outputs, final_state = tf.compat.v1.nn.dynamic_rnn(g_cell, Z, dtype=tf.float32, sequence_length=T)
                E = _layers.fully_connected(g_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)

            return E

        def supervisor(H, T):
            """Generate next sequence using the previous sequence.

      Args:
        - H: latent representation
        - T: input time information

      Returns:
        - S: generated sequence based on the latent representations generated by the generator
      """
            with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
                s_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                    [rnn_cell(module_name, hidden_dim) for _ in range(num_layers - 1)])
                s_outputs, final_state = tf.compat.v1.nn.dynamic_rnn(s_cell, H, dtype=tf.float32, sequence_length=T)
                S = _layers.fully_connected(s_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)

            return S

        def discriminator(H, T):
            """Discriminate the original and synthetic time-series data.

      Args:
        - H: latent representation
        - T: input time information

      Returns:
        - Y_hat: classification results between original and synthetic time-series
      """
            with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
                d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                    [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
                d_outputs, final_state = tf.compat.v1.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length=T)
                Y_hat = _layers.fully_connected(d_outputs, 1, activation_fn=None)

            return Y_hat

        # Embedder & Recovery

        H = embedder(self.X, self.T)
        X_tilde = recovery(H, self.T)

        # Generator
        E_hat = generator(self.Z, self.T)
        H_hat = supervisor(E_hat, self.T)
        H_hat_supervise = supervisor(H, self.T)

        # Synthetic data
        self.X_hat = recovery(H_hat, self.T)

        # Discriminator
        Y_fake = discriminator(H_hat, self.T)
        Y_real = discriminator(H, self.T)
        Y_fake_e = discriminator(E_hat, self.T)

        # Variables
        e_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('embedder')]
        r_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('recovery')]
        g_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('generator')]
        s_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('supervisor')]
        d_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]

        # Discriminator loss
        D_loss_real = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
        D_loss_fake = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
        D_loss_fake_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        # Generator loss
        # 1. Adversarial loss
        G_loss_U = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
        G_loss_U_e = tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)

        # 2. Supervised loss
        G_loss_S = tf.compat.v1.losses.mean_squared_error(H[:, 1:, :], H_hat_supervise[:, :-1, :])

        # 3. Two Momments
        G_loss_V1 = tf.reduce_mean(
            tf.abs(tf.sqrt(tf.nn.moments(self.X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(self.X, [0])[1] + 1e-6)))
        G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(self.X_hat, [0])[0]) - (tf.nn.moments(self.X, [0])[0])))

        G_loss_V = G_loss_V1 + G_loss_V2

        # 4. Summation
        G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V

        # Embedder network loss
        E_loss_T0 = tf.compat.v1.losses.mean_squared_error(self.X, X_tilde)
        E_loss0 = 10 * tf.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1 * G_loss_S

        # optimizer
        E0_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
        E_solver = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
        D_solver = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
        G_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)
        GS_solver = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)

        saver = tf.compat.v1.train.Saver()

        ## TimeGAN training
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        losses = dict()
        times = dict()
        # all_weights = dict()

        start_time_train = datetime.now()

        e_losses = list()
        d_losses = list()
        s_losses = list()
        g_losses_u = list()
        g_losses_s = list()
        g_losses_v = list()
        e_losses_t0 = list()

        # 1. Embedding network training
        print('Start Embedding Network Training')

        start_time_embedding = datetime.now()

        for itt in tqdm(range(iterations_embedding), desc='Embedding Network Training'):
            # Set mini-batch
            X_mb, T_mb = batch_generator(self.ori_data, self.ori_time, batch_size)
            # Train embedder
            _, step_e_loss = self.sess.run([E0_solver, E_loss_T0], feed_dict={self.X: X_mb, self.T: T_mb})
            e_losses.append(np.round(np.sqrt(step_e_loss), 4))

        end_time_embedding = datetime.now()
        print('Time elapsed total (hh:mm:ss.ms) {}'.format(end_time_embedding - start_time_embedding))

        print('Finish Embedding Network Training')

        # 2. Training only with supervised loss
        print('Start Training with Supervised Loss Only')

        start_time_supervised = datetime.now()

        for itt in tqdm(range(iterations_supervised), desc='Supervised Loss Network Training'):
            # Set mini-batch
            X_mb, T_mb = batch_generator(self.ori_data, self.ori_time, batch_size)
            # Random vector generation
            Z_mb = random_generator(batch_size, self.z_dim, T_mb, self.max_seq_len)
            # Train generator
            _, step_g_loss_s = self.sess.run([GS_solver, G_loss_S], feed_dict={self.Z: Z_mb, self.X: X_mb, self.T: T_mb})
            s_losses.append(np.round(np.sqrt(step_g_loss_s), 4))

        end_time_supervised = datetime.now()
        print('Time elapsed supervised training (hh:mm:ss.ms) {}'.format(end_time_supervised - start_time_supervised))
        print('Finish Training with Supervised Loss Only')

        # 3. Joint Training
        print('Start Joint Training')

        start_time_joint = datetime.now()

        for itt in tqdm(range(iterations_joint), desc='Joint Training'):
            # Generator training (twice more than discriminator training)
            for kk in range(2):
                # Set mini-batch
                X_mb, T_mb = batch_generator(self.ori_data, self.ori_time, batch_size)
                # Random vector generation
                Z_mb = random_generator(batch_size, self.z_dim, T_mb, self.max_seq_len)
                # Train generator
                _, step_g_loss_u, step_g_loss_s, step_g_loss_v = self.sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V],
                                                                          feed_dict={self.Z: Z_mb, self.X: X_mb, self.T: T_mb})
                # Train embedder
                _, step_e_loss_t0 = self.sess.run([E_solver, E_loss_T0], feed_dict={self.Z: Z_mb, self.X: X_mb, self.T: T_mb})

                # Discriminator training
            # Set mini-batch
            X_mb, T_mb = batch_generator(self.ori_data, self.ori_time, batch_size)
            # Random vector generation
            Z_mb = random_generator(batch_size, self.z_dim, T_mb, self.max_seq_len)
            # Check discriminator loss before updating
            check_d_loss = self.sess.run(D_loss, feed_dict={self.X: X_mb, self.T: T_mb, self.Z: Z_mb})
            d_loss = check_d_loss
            # Train discriminator (only when the discriminator does not work well)
            if (check_d_loss > 0.15):
                _, step_d_loss = self.sess.run([D_solver, D_loss], feed_dict={self.X: X_mb, self.T: T_mb, self.Z: Z_mb})
                d_loss = step_d_loss

            d_losses.append(np.round(d_loss, 4))
            g_losses_u.append(np.round(step_g_loss_u, 4))
            g_losses_v.append(np.round(step_g_loss_v, 4))
            g_losses_s.append(np.round(np.sqrt(step_g_loss_s), 4))
            e_losses_t0.append(np.round(np.sqrt(step_e_loss_t0), 4))

        end_time_joint = datetime.now()
        print('Time elapsed joint training (hh:mm:ss.ms) {}'.format(end_time_joint - start_time_joint))
        end_time_train = datetime.now()
        print('Time elapsed to totally train (hh:mm:ss.ms) {}'.format(end_time_train - start_time_train))

        print('Finish Joint Training')

        losses['s_loss'] = s_losses
        losses['e_loss'] = e_losses
        losses['d_loss'] = d_losses
        losses['g_loss_u'] = g_losses_u
        losses['g_loss_s'] = g_losses_s
        losses['g_loss_v'] = g_losses_v
        losses['e_loss_t0'] = e_losses_t0

        times['embedding'] = str(end_time_embedding - start_time_embedding)
        times['supervised'] = str(end_time_supervised - start_time_supervised)
        times['joint'] = str(end_time_joint - start_time_joint)
        times['train'] = str(end_time_train - start_time_train)

        # Synthetic data generation

        start_time_generate = datetime.now()
        #
        #
        end_time_generate = datetime.now()
        #
        #
        end_time_total = datetime.now()
        #
        #
        times['generate'] = str(end_time_generate - start_time_generate)
        times['total'] = str(end_time_total - start_time_total)


        # save the model
        output_path = os.path.join(self.parameters['output_path'], self.parameters['dgp_type'], str(self.parameters['run_id']))
        if os.path.exists(output_path):
            print('[INFO] Saving the synthesizer...')
            save_path = saver.save(self.sess, os.path.join(output_path, f'model_{self.parameters["run_id"]}.ckpt'))
            print(f"Model saved in path: {save_path}")
        else:
            raise ValueError(f'Output path {output_path} does not exist')

        return losses, times
