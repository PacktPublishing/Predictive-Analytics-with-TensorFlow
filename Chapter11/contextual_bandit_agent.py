import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os

from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

class contextualBandit():
    def __init__(self):
        '''
        This constructor, lists out all of our bandits. We assume the current state being arms 4, 2, 3 and 1 that are the most optimal respectively
        '''
        self.state = 0        
        self.bandits = np.array([[0.2,0,-0.0,-5], [0.1,-5,1,0.25], [0.3,0.4,-5,0.5], [-5,5,5,5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]
        
    def getBandit(self):        
        '''
        This function returns a random state for each episode.
        '''
        self.state = np.random.randint(0, len(self.bandits)) 
        return self.state
        
    def pullArm(self,action):        
        '''
        This funciton creates the reword to the bandits on the basis of randomly generated numbers. It then returns either a positive or negative reward -i.e. action
        ''' 
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1

class ContextualAgent():
    def __init__(self, lr, s_size,a_size):
        '''
        This function establishes the feed-forward part of the network. The agent takes a state and produces an action -i.e. contextual agent
        ''' 
        self.state_in= tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, s_size)
        output = slim.fully_connected(state_in_OH, a_size,biases_initializer=None, activation_fn=tf.nn.sigmoid, weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(self.output,0)
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder,[1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

tf.reset_default_graph() #Clear the Tensorflow graph.
lrarning_rate = 0.001
contextualBandit = contextualBandit() #Load the bandits.
contextualAgent = ContextualAgent(lr=lrarning_rate, s_size=contextualBandit.num_bandits, a_size=contextualBandit.num_actions) #Load the agent.
weights = tf.trainable_variables()[0] #The weights we will evaluate to look into the network.

max_iteration = 10000 #Set the max iteration for training the agent.
total_reward = np.zeros([contextualBandit.num_bandits,contextualBandit.num_actions]) #Set scoreboard for bandits to 0.
chance_of_random_action = 0.1 #Set the chance of taking a random action.

init_op = tf.global_variables_initializer()
right_flag = 0
wrong_flag = 0

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init_op)
    i = 0
    while i < max_iteration:
        s = contextualBandit.getBandit() #Get a state from the environment.
        #Choose either a random action or one from our network.
        if np.random.rand(1) < chance_of_random_action:
            action = np.random.randint(contextualBandit.num_actions)
        else:
            action = sess.run(contextualAgent.chosen_action,feed_dict={contextualAgent.state_in:[s]})
        reward = contextualBandit.pullArm(action) #Get our reward for taking an action given a bandit.
        #Update the network.
        feed_dict={contextualAgent.reward_holder:[reward],contextualAgent.action_holder:[action],contextualAgent.state_in:[s]}
        _,ww = sess.run([contextualAgent.update,weights], feed_dict=feed_dict)        
        #Update our running tally of scores.
        total_reward[s,action] += reward
        if i % 500 == 0:
            print("Mean reward for each of the " + str(contextualBandit.num_bandits) + " bandits: " + str(np.mean(total_reward,axis=1)))
        i+=1

for a in range(contextualBandit.num_bandits):
    print("The agent thinks action " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " would be the most efficient one.")
    if np.argmax(ww[a]) == np.argmin(contextualBandit.bandits[a]):
        right_flag += 1
        print(" and it was right at the end!")
    else:
        print(" and it was wrong at the end!")
        wrong_flag += 1

prediction_accuracy = (right_flag/(right_flag+wrong_flag))
print("Prediction accuracy (%):", prediction_accuracy * 100)
