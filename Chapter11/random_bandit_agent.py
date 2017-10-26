import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os

from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

bandits = [0.5,0,-0.5,-8]
num_bandits = len(bandits)

def getBandit(bandit):

    '''
    This funciton creates the reword to the bandits on the basis of 
    randomly generated numbers. It then returns either a positive or negative reward.
    ''' 
    random_number = np.random.randn(1)
    if random_number > bandit:   
        return 1
    else:
        return -1

# Reset the default tensorflow graph
tf.reset_default_graph()
weight_op = tf.Variable(tf.ones([num_bandits]))
action_op = tf.argmax(weight_op,0)

reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(weight_op,action_holder,[1])

#Define the objective funciton
loss = -(tf.log(responsible_weight)*reward_holder)

LR = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
training_op = optimizer.minimize(loss)

# Total number of iteration to traing the agent
total_episodes = 10000
total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
chance_of_random_action = 0.1 #Set the probablity of taking a random action.

#Initialize the global variables
init_op = tf.global_variables_initializer()

# Create a tensorflow sesion and launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init_op)
    i = 0
    while i < total_episodes:
        
        #Choose either a random action or one from the network.
        if np.random.rand(1) < chance_of_random_action:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(action_op)
        
        reward = getBandit(bandits[action]) #Get our reward from picking one of the bandits.
        
        #Update the network.
        _,resp,ww = sess.run([training_op,responsible_weight,weight_op], feed_dict={reward_holder:[reward],action_holder:[action]})
        
        #Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for all the " + str(num_bandits) + " bandits: " + str(total_reward))
        i+=1

print("The agent thinks bandit " + str(np.argmax(ww)+1) + " would be the most efficient one.")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print(" and it was right at the end!")
else:
    print(" and it was wrong at the end!")


