import os

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
# OpenCV2 for saving an image
import cv2

import tensorflow as tf
import numpy as np

import pickle

from ops import conv2d, dense


IMG_WIDTH = 75
IMG_HEIGHT = 75
CHANNELS = 2
OUTPUT_SIZE = 3

NUM_EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
DROPOUT_KEEP_PROBABILITY = 0.5

FILENAME_TRAINING_DATA = "data_training"
FILENAME_TEST_DATA = "data_test"

SAVE_PATH = "./cv_graph/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

class TrainingData:
    def __init__(self):
        self.img = []
        self.coordinate = []
        self.orientation = []

    def add(self, im, c, o):
        self.img.append(im)
        self.coordinate.append(c)
        self.orientation.append(o)


class CVGraph:
    def __init__(self, h, w, c, o):
        self.screen_height = h
        self.screen_width = w
        self.channels = c
        self.output_size = o
        self.orientation_size = 4
        self.cnn_output_size = 3

    def buildGraph(self):
        with tf.variable_scope("CV_graph"):
            self.input = tf.placeholder('float32',
                [None, self.screen_height, self.screen_width, self.channels], name='input')

            initializer = tf.contrib.layers.xavier_initializer()
            activation_fn = tf.nn.relu

            self.conv1, self.conv1_w, self.conv1_b = conv2d(self.input,
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


            self.keep_probability = tf.placeholder('float32', name='keep_probability')

            self.dense1, self.dense1_w, self.dense1_b = dense(inputs=self.flat, units=100,
                                                                       activation=tf.nn.relu, name='dense1')
            self.dropout1 = tf.nn.dropout(self.dense1, self.keep_probability)
            self.dense2, self.dense2_w, self.dense2_b = dense(inputs=self.dropout1, units=50,
                                                                       activation=tf.nn.relu, name='dense2')
            self.dropout2 = tf.nn.dropout(self.dense2, self.keep_probability)

            self.dense3, self.dense3_w, self.dense3_b = dense(inputs=self.dropout2,
                                                                       units=10, activation=tf.nn.relu,
                                                                       name='dense3')

            self.output, self.output_w, self.output_b = dense(inputs=self.dense3,
                                                              units=self.output_size, activation=None,
                                                              name='output')

            self.ground_truth = tf.placeholder('float32', [None, self.output_size], name='ground_truth')
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.output, self.ground_truth)))
            self.global_step = tf.Variable(0, trainable=False)

            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

    def train(self, sess, img, labels, step):
        _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.input: img,
                                                                   self.ground_truth: labels,
                                                                   self.global_step: step,
                                                                   self.keep_probability: DROPOUT_KEEP_PROBABILITY})
        return loss

    def prediction(self, sess, img):
        output = sess.run(self.output, feed_dict={self.input: img,
                                                  self.keep_probability: 1.0})

        return output

    def validate(self, sess, img, labels, step):
        loss = sess.run(self.loss, feed_dict={self.input: img,
                                              self.ground_truth: labels,
                                              self.global_step: step,
                                              self.keep_probability: 1.0})
        return loss


def convertImage(img):
    bridge = CvBridge()

    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img, "mono8")
    except CvBridgeError, e:
        print(e)

    cv2_img = cv2.resize(cv2_img, (IMG_WIDTH, IMG_HEIGHT))

    return cv2_img

def merge(image, label, old_image, old_label):
    merged_image = np.concatenate((np.expand_dims(image, -1), np.expand_dims(old_image, -1)), axis=2)
    merged_label = label - old_label
    return merged_image, merged_label #return 2 channel image consisting of 2 states

def process(data):
    for i in range(len(data.img)):
        if (i+1 >= len(data.img)):
            merged_image, merged_label = merge(data.img[i], data.coordinate[i], data.img[i], data.coordinate[i])
            data.img[i] = merged_image
            data.coordinate[i] = merged_label
        else:
            merged_image, merged_label = merge(data.img[i+1], data.coordinate[i+1], data.img[i], data.coordinate[i])
            data.img[i] = merged_image
            data.coordinate[i] = merged_label

    return data

def sampleData(data, index, sample_size):
    indexes = range(index, index + sample_size)
    images = np.array([data.img[i] for i in indexes])
    labels = np.array([[float(data.coordinate[i][0]), float(data.coordinate[i][1]), float(data.coordinate[i][2])] for i in indexes])

    return images, labels


def visualize(data):

    for i in range(len(data.img)):
        print i, data.coordinate[i], data.orientation[i]
        plt.imshow(data.img[i], 'gray'), plt.title(str(i))
        plt.show()


def train_net():
    global sess
    sess = tf.Session()

    graph = CVGraph(IMG_HEIGHT, IMG_WIDTH, CHANNELS, OUTPUT_SIZE)
    graph.buildGraph()

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()

    #data containing current camera image, distance to target, orientation
    data = pickle.load(open(FILENAME_TRAINING_DATA, "rb"))
    test_data = pickle.load(open(FILENAME_TEST_DATA, "rb"))

    data = process(data)
    test_data = process(test_data)

    global_step = 1

    for epoch in range(NUM_EPOCHS):
        print "epoch ", epoch
        avg_loss = 0
        num_batches = int(len(data.img)/(BATCH_SIZE))
        index = 0
        for i in range(num_batches):
            images, labels = sampleData(data, index, BATCH_SIZE)
            loss = graph.train(sess, images, labels, global_step)
            avg_loss += loss
            global_step += 1
            index += BATCH_SIZE

        avg_loss = avg_loss / float(num_batches)

        saver.save(sess, SAVE_PATH + 'cv_graph.cptk')
        test_loss = test_net(sess, graph, global_step, test_data)

        print "validation Loss: ", test_loss
        print "training Loss: ", avg_loss


def test_net(sess, graph, global_step, test_data):
    num_batches = int(len(test_data.img) / BATCH_SIZE)
    index = 0
    loss = 0
    for i in range(num_batches):
        images, labels = sampleData(test_data, index, BATCH_SIZE)
        loss += graph.validate(sess, images, labels, global_step)
        index += BATCH_SIZE

    return loss/num_batches


if __name__ == '__main__':
    print("starting...")
    train_net()
