#!/usr/bin/env python

import numpy as np

class History:
    def __init__(self, actionNumber, actionSize):
        self.states = []
        self.actions = []
        self.rewards = []
        self.responses = []
        self.actionNumber = actionNumber
        self.actionSize = actionSize

    def addAction2History(self, action):

        self.actions.append(action)

    def getActions(self):
        return self.actions

    def getLastAction(self):
        assert (len(self.actions) > 0), "Action history is empty!"
        return self.actions[-1]

    def addState2History(self, state):
        self.states.append(state)

    def addResponse2History(self, response):
        self.responses.append(response)

    def getStates(self):
        return self.states[:-1]

    def getResponses(self):
        return self.responses[:-1]

    def getState(self, iterator):
        assert (len(self.states) > 0), "State history is empty!"
        return np.expand_dims(self.states[iterator], 0)

    def getLastState(self):
        assert (len(self.states) > 0), "State history is empty!"
        return np.expand_dims(self.states[-1], 0)

    def addReward2History(self, reward):
        self.rewards.append(reward)

    def getRewardHistory(self):
        return self.rewards

    def getLastReward(self):
        assert (len(self.rewards) > 0), "Reward history is empty!"
        return self.rewards[-1]

    def sumRewards(self):
        return sum(self.rewards)

    def clean(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.responses = []