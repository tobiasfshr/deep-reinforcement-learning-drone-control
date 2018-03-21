#!/usr/bin/env python

import rospy
from rotors_reinforce.srv import PerformAction
from rotors_reinforce.srv import GetState
from History import *
from model import ComputationalGraph

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

import tensorflow as tf
import random
import numpy as np

TRAIN_MODEL = 1

EPISODE_LENGTH = 120

EPSILON = 0.9
EPSILON_DECAY = 0.97
EPSILON_MIN = 0.01
EPSILON_STEP = 10

ACTION_NUM = 3
ACTION_SIZE = 3
IMG_HEIGHT = 75
IMG_WIDTH = 75
IMG_CHANNELS = 2
RESPAWN_CODE = [0, 0, 0, 42]

bridge = CvBridge()


def convertImage(img):
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(img, "mono8")
    except CvBridgeError, e:
        print(e)

    # Format for the Tensor
    cv2_img = cv2.resize(cv2_img, (IMG_WIDTH, IMG_HEIGHT))

    return cv2_img


def convertState(state, old_state):
    converted_state = np.concatenate((np.expand_dims(convertImage(state.img), -1), np.expand_dims(convertImage(old_state.img), -1)), axis=2)
    return converted_state #return 2 channel image consisting of 2 states
    #return np.expand_dims(convertImage(state.img), -1)

def chooseAction(probabilities, e, is_training):
    action = np.zeros(ACTION_NUM)

    for i in range(len(probabilities[0])):

        if is_training and random.uniform(0, 1) < e:
            action[i] = random.randint(0, 2)
        else:
            action[i] = np.argmax(probabilities[0][i])

    return action



def reinforce_node():

    #set up env
    rospy.init_node('ReinforceLearning', anonymous=True)
    perform_action_client = rospy.ServiceProxy('env_tr_perform_action', PerformAction)
    get_state_client = rospy.ServiceProxy('env_tr_get_state', GetState)

    history = History(ACTION_NUM, ACTION_SIZE)

    graph = ComputationalGraph(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    action = np.zeros(ACTION_NUM)
    executed_action = np.zeros(ACTION_NUM + 1)

    sess = tf.Session()

    graph.constructGraph(sess, ACTION_NUM, ACTION_SIZE)

    # restoring agent
    saver = tf.train.Saver()
    try:
        saver.restore(sess, "./log/model.ckpt")
        print "model restored"
    except:
        print "model restore failed. random initialization"
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # restore pretrained layers for random init
        vars  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CV_graph')
        cv_saver = tf.train.Saver(vars)
        cv_saver.restore(sess, "./cv_graph/cv_graph.cptk")
        print "pretrained weights restored"


    step = 0
    e = EPSILON

    # main loop:
    while not rospy.is_shutdown():
        crashed_flag = False

        #get initial state
        rospy.wait_for_service('env_tr_get_state')
        try:
            response = get_state_client()
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e


        old_response = response

        state = convertState(response, old_response)
        history.clean()
        history.addState2History(state)

        print response.target_position

        episode_step = 0
        # run episode, while not crashed and simulation is running
        while not crashed_flag and not rospy.is_shutdown():
            #get most probable variant to act for each action, and the probabilities
            probabilities = graph.calculateAction(sess, history.getLastState())

            #choose action according to softmax distribution (add epsilon randomness in training)
            action = chooseAction(probabilities, e, TRAIN_MODEL)

            #choose roll, pitch and thrust according to network output
            executed_action[0] = (float(action[0]) - float(ACTION_SIZE) / 2) * 2.0 / float(ACTION_SIZE - 1)
            executed_action[1] = (float(action[1]) - float(ACTION_SIZE) / 2) * 2.0 / float(ACTION_SIZE - 1)
            #executed_action[2] = (float(action[2]) - float(ACTION_SIZE) / 2)
            executed_action[2] = 0. #we skip the yaw command dimesion
            executed_action[3] = float(action[2]) / float(ACTION_SIZE - 1)

            rospy.wait_for_service('env_tr_perform_action')
            try:
                response = perform_action_client(executed_action)
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e

            state = convertState(response, old_response)
            old_response = response

            #update history
            actionmatrix = np.zeros([ACTION_NUM, ACTION_SIZE])
            for i in xrange(len(actionmatrix)):
                actionmatrix[i][int(action[i])] = 1

            history.addAction2History(actionmatrix)
            history.addState2History(state)
            history.addResponse2History(response)
            history.addReward2History(response.reward)

            crashed_flag = response.crashed

            episode_step+=1
            if episode_step >= EPISODE_LENGTH:
                rospy.wait_for_service('env_tr_perform_action')
                try:
                    response = perform_action_client(RESPAWN_CODE)
                except rospy.ServiceException, e:
                    print "Service call failed: %s" % e
                break

        # update policy
        if TRAIN_MODEL == 1:
            graph.updatePolicy(sess, history, step)

        #save every 50 episodes
        if TRAIN_MODEL == 1 and step % 50 == 0:
            save_path = saver.save(sess, "log/model.ckpt")
            print("Model saved in file: %s" % save_path)

        if step % EPSILON_STEP == 0:
            e = e * EPSILON_DECAY

        print "episode number: ", step
        print "total reward: ", history.sumRewards()
        step += 1


if __name__ == '__main__':
    try:
        print("starting...")
        reinforce_node()
    except rospy.ROSInterruptException:
        pass

