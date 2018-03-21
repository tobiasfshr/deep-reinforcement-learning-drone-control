import tensorflow as tf
import numpy as np
from ops import conv2d, dense, variable_summaries

class ComputationalGraph:
    def __init__(self, img_h, img_w, img_c):
        self.image_height, self.image_width, self.image_channels = img_h, img_w, img_c

        #learning rates suggested in A3C paper
        self.lrate = 2.5e-3
        self.learning_rate_minimum = 5.0e-4

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_step = 40
        self.dropout_keep_prob = 0.5

    def constructGraph(self, sess, action_num, action_size):
        self.episode_rewards = tf.placeholder(tf.float32, [None], name="episode_rewards")
        self.state = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_channels], name="state")

            ##ConvNet for feature extraction
        with tf.variable_scope("CV_graph"):

            initializer = tf.contrib.layers.xavier_initializer()
            activation_fn = tf.nn.relu

            self.conv1, self.conv1_w, self.conv1_b = conv2d(self.state,
                                                            16,
                                                            [5, 5],
                                                            [1, 1],
                                                            initializer,
                                                            activation_fn,
                                                            'NHWC',
                                                            name='conv1')


            self.conv2, self.conv2_w, self.conv2_b = conv2d(self.conv1,
                                                            16,
                                                            [5, 5],
                                                            [1, 1],
                                                            initializer,
                                                            activation_fn,
                                                            'NHWC',
                                                            name='conv2')

            self.max_pool1 = tf.nn.max_pool(self.conv2,
                                            [1, 2, 2, 1],
                                            [1, 2, 2, 1],
                                            padding='VALID',
                                            name='max_pool1')

            self.conv3, self.conv3_w, self.conv3_b = conv2d(self.max_pool1,
                                                            32,
                                                            [3, 3],
                                                            [1, 1],
                                                            initializer,
                                                            activation_fn,
                                                            'NHWC',
                                                            name='conv3')

            self.conv4, self.conv4_w, self.conv4_b = conv2d(self.conv3,
                                                            32,
                                                            [3, 3],
                                                            [1, 1],
                                                            initializer,
                                                            activation_fn,
                                                            'NHWC',
                                                            name='conv4')

            self.max_pool2 = tf.nn.max_pool(self.conv4,
                                            [1, 2, 2, 1],
                                            [1, 2, 2, 1],
                                            padding='VALID',
                                            name='max_pool2')

            self.flat = tf.contrib.layers.flatten(self.max_pool2)

            self.dense1, self.dense1_w, self.dense1_b = dense(inputs=self.flat, units=100,
                                                                       activation=tf.nn.relu, name='dense1')
            self.dropout1 = tf.nn.dropout(self.dense1, self.dropout_keep_prob)
            self.dense2, self.dense2_w, self.dense2_b = dense(inputs=self.dropout1, units=50,
                                                                       activation=tf.nn.relu, name='dense2')
            self.dropout2 = tf.nn.dropout(self.dense2, self.dropout_keep_prob)

        with tf.variable_scope("Agent"):
            ##policy network

            self.po_dense3, self.po_dense3_w, self.po_dense3_b = dense(inputs=self.dropout2, units=10,
                                                                    activation=tf.nn.relu, name='po_dense3')

            self.po_dense4, self.po_dense4_w, self.po_dense4_b = dense(inputs=self.po_dense3, units=action_num*action_size, activation=None, name='po_dense4')
            self.po_probabilities = tf.nn.softmax(tf.reshape(self.po_dense4, [-1, action_num, action_size]))

            self.po_prev_actions = tf.placeholder(tf.float32, [None, action_num, action_size], name="po_prev_action")
            self.po_return = tf.placeholder(tf.float32, [None, 1], name="po_return")
            self.po_eligibility = tf.log(tf.reduce_sum(tf.multiply(self.po_prev_actions, self.po_probabilities), axis=-1)) * self.po_return
            self.po_loss = -tf.reduce_sum(self.po_eligibility)

            ##value network

            self.v_dense3, self.v_dense3_w, self.v_dense3_b = dense(inputs=self.dropout2, units=10, activation=tf.nn.relu, name='v_dense3')

            self.v_output, self.v_output_w, self.v_output_b = dense(inputs=self.v_dense3, activation=None, units=1, name='v_output')

            self.v_actual_return = tf.placeholder(tf.float32, [None, 1], name="v_actual_return")
            self.v_loss = tf.nn.l2_loss(tf.subtract(self.v_output, self.v_actual_return))

            ##optimization

            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                                 tf.train.exponential_decay(
                                                     self.lrate,
                                                     self.global_step,
                                                     self.decay_step,
                                                     0.98,
                                                     staircase=True))

            self.loss = 0.5 * self.v_loss + self.po_loss

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate_op).minimize(self.loss)

        self.constructSummary(sess)

    def constructSummary(self, sess):
        variable_summaries(self.episode_rewards)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./log/train', sess.graph)


    def calculateAction(self, sess, state):
        return sess.run(self.po_probabilities, feed_dict={self.state: state})

    def calculateReward(self, sess, state):
        reward = sess.run(self.v_output, feed_dict={self.state: state})
        return reward[0][0]

    def updatePolicy(self, sess, history, step):
        rewards = history.getRewardHistory()
        advantages = []
        update_vals = []
        episode_reward = 0

        for i, reward in enumerate(rewards):
            episode_reward += reward
            future_reward = 0
            future_transitions = len(rewards) - i
            decrease = 1
            for index2 in xrange(future_transitions):
                future_reward += rewards[(index2) + i] * decrease
                decrease = decrease * 0.98

            prediction = self.calculateReward(sess, history.getState(i))
            advantages.append(future_reward - prediction)
            update_vals.append(future_reward)

        statistics, _ = sess.run([self.merged, self.optimizer], feed_dict={self.state: history.getStates(),
                                                    self.po_prev_actions: history.getActions(),
                                                    self.po_return: np.expand_dims(advantages, axis=1),
                                                    self.episode_rewards: history.getRewardHistory(),
                                                    self.v_actual_return: np.expand_dims(update_vals, axis=1),
                                                    self.global_step: step
                                                    })

        self.train_writer.add_summary(statistics, step)
        self.train_writer.flush()