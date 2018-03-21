# deep-reinforcement-learning-drone-control
This is a deep reinforcement learning based drone control system implemented in python (Tensorflow/ROS) and C++ (ROS). To test it, please clone the rotors simulator from https://github.com/ethz-asl/rotors_simulator in your catkin workspace. Copy the multirotor_base.xarco to the rotors simulator for adding the camera to the drone.

The drone control system operates on camera images as input and a discretized version of the steering commands as output. The neural network model is end-to-end and a non-asynchronous implementation of the A3C model (https://arxiv.org/pdf/1602.01783.pdf), because the gazebo simulator is not capable of running multiple copies in parallel (and neither is my laptop :D). The training is performed on the basis of pretrained weights from a supervised learning task, since the simulator is very resource intensive and training is time consuming.

The outcome was discussed within a practical course at the RWTH Aachen, where this agent served as a proof-of-concept, that it is possible to efficiently train an end-to-end deep reinforcement learning model on the task of controlling a drone in a realistic 3D environment.
