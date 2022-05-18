import gym
import tensorflow.compat.v1 as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 指定gpu编号，从0开始
tf.disable_v2_behavior()
import random
from collections import deque
import numpy as np
import os

# experiences replay buffer size
REPLAY_SIZE = 2000
# memory size 1000
# size of minibatch
small_BATCH_SIZE = 16
big_BATCH_SIZE = 128
BATCH_SIZE_door = 1000

# these are the hyper Parameters for DQN
# discount factor for target Q to caculate the TD aim value
GAMMA = 0.9
# the start value of epsilon E
INITIAL_EPSILON = 0.5
# the final value of epsilon
FINAL_EPSILON = 0.01

class DQN():
    def __init__(self, observation_width, observation_height, action_space, model_file , log_file):
        # the state is the input vector of network, in this env, it has four dimensions
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        # the action is the output vector and it has two dimensions
        self.action_dim = action_space
        # init experience replay, the deque is a list that first-in & first-out
        self.replay_buffer = deque()
        self.replay_buffer_defender = deque()
        # you can create the network by the two parameters
        self.create_Q_network()
        self.create_Q_network_defender()
        # after create the network, we can define the training methods
        self.create_updating_method()
        self.create_updating_method_defender()
        # set the value in choose_action
        self.epsilon = INITIAL_EPSILON
        self.model_path = model_file + "/save_model.ckpt"
        self.model_path_defender = model_file + "_defender" + "/save_model.ckpt"
        self.model_file = model_file
        self.model_file_defender = model_file + "_defender"
        self.log_file = log_file
        self.log_file_defender = "defender_" + log_file
        # 因为保存的模型名字不太一样，只能检查路径是否存在
        # Init session
        self.sess = tf.InteractiveSession()

        # self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        # self.sess1 = tf.InteractiveSession(config=self.config)
        # self.sess2 = tf.Session(config=self.config)

        if os.path.exists(self.model_file):
            print("model exists , load model\n")
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.model_path)
            # print("=========All Variables==========")
            # print_tensors_in_checkpoint_file(self.model_path, tensor_name=None,
            #                                  all_tensors=True, all_tensor_names=True)
        else:
            print("model don't exists , create new one\n")
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()


        if os.path.exists(self.model_file_defender):
            print("defender model exists , load model\n")
            self.saver_defender = tf.train.Saver()
            self.saver_defender.restore(self.sess, self.model_path_defender)
            # print("=========All Variables==========")
            # print_tensors_in_checkpoint_file(self.model_path, tensor_name=None,
            #                                  all_tensors=True, all_tensor_names=True)
        else:
            print("defender model don't exists , create new one\n")
            self.sess.run(tf.global_variables_initializer())
            self.saver_defender = tf.train.Saver()

        # init
        # 只有把框图保存到文件中，才能加载到浏览器中观看
        self.writer = tf.summary.FileWriter(self.log_file, self.sess.graph)
        ####### 路径中不要有中文字符，否则加载不进来 ###########
        # tensorboard --logdir=logs_gpu --host=127.0.0.1
        self.merged = tf.summary.merge_all()
        # 把所有summary合并在一起，就是把所有loss,w,b这些的数据打包
        # 注意merged也需要被sess.run才能发挥作用

        self.writer_defender = tf.summary.FileWriter(self.log_file_defender, self.sess.graph)
        ####### 路径中不要有中文字符，否则加载不进来 ###########
        # tensorboard --logdir=logs_gpu --host=127.0.0.1
        self.merged_defender = tf.summary.merge_all()
        
        
        
    # the function that give the weight initial value
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # the function that give the bias initial value
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
        # stride第一个和最后一个元素一定要为1，中间两个分别是x和y轴的跨度，此处设为1
        # SAME 抽取时外面有填充，抽取大小是一样的
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # the function to create the network
    # there are two networks, the one is action_value and the other is target_action_value
    # these two networks has same architecture
    def create_Q_network(self):
        with tf.name_scope('inputs'):
            # first, set the input of networks
            self.state_input = tf.placeholder("float", [None, self.state_h, self.state_w, 15])
        # second, create the current_net
        with tf.variable_scope('current_net'):
            # first, set the network's weights
            W_conv1 = self.weight_variable([5,5,15,32])
            b_conv1 = self.bias_variable([32])
            W_conv2 = self.weight_variable([5,5,32,64])
            b_conv2 = self.bias_variable([64])
            # W_conv3 = self.weight_variable([5,5,64,128])
            # b_conv3 = self.bias_variable([128])
            W1 = self.weight_variable([int((self.state_w/4) * (self.state_h/4) * 64), 512])
            b1 = self.bias_variable([512])
            W2 = self.weight_variable([512, 256])
            b2 = self.bias_variable([256])
            W3 = self.weight_variable([256, self.action_dim])
            b3 = self.bias_variable([self.action_dim])
            # second, set the layers
            # conv layer one
            h_conv1 = tf.nn.relu(self.conv2d(self.state_input, W_conv1) + b_conv1)
            # self.state_w * self.state_h * 32
            # pooling layer one
            h_pool1 = self.max_pool_2x2(h_conv1)
            # self.state_w/2 * self.state_h/2 * 32
            # conv layer two
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            # pooling layer two
            h_pool2 = self.max_pool_2x2(h_conv2)
            # self.state_w/4 * self.state_h/4 * 64
            # conv layer three
            # h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
            # self.state_w/4 * self.state_h/4 * 128
            h_conv2_flat = tf.reshape(h_pool2, [-1,int((self.state_w/4) * (self.state_h/4) * 64)])
            # hidden layer one
            h_layer_one = tf.nn.relu(tf.matmul(h_conv2_flat, W1) + b1)
            # dropout
            h_layer_one = tf.nn.dropout(h_layer_one, 1)
            # hidden layer two
            h_layer_two = tf.nn.relu(tf.matmul(h_layer_one, W2) + b2)
            # dropout
            h_layer_two = tf.nn.dropout(h_layer_two, 1)
            # the output of current_net
            Q_value = tf.matmul(h_layer_two, W3) + b3
            # dropout
            self.Q_value = tf.nn.dropout(Q_value, 1)
        # third, create the current_net
        with tf.variable_scope('target_net'):
            # first, set the network's weights
            t_W_conv1 = self.weight_variable([5,5,15,32])
            t_b_conv1 = self.bias_variable([32])
            t_W_conv2 = self.weight_variable([5,5,32,64])
            t_b_conv2 = self.bias_variable([64])
            # t_W_conv3 = self.weight_variable([5,5,64,128])
            # t_b_conv3 = self.bias_variable([128])
            t_W1 = self.weight_variable([int((self.state_w/4) * (self.state_h/4) * 64), 512])
            t_b1 = self.bias_variable([512])
            t_W2 = self.weight_variable([512, 256])
            t_b2 = self.bias_variable([256])
            t_W3 = self.weight_variable([256, self.action_dim])
            t_b3 = self.bias_variable([self.action_dim])
            # second, set the layers
            # conv layer one
            t_h_conv1 = tf.nn.relu(self.conv2d(self.state_input, t_W_conv1) + t_b_conv1)
            # self.state_w * self.state_h * 32
            # pooling layer one
            t_h_pool1 = self.max_pool_2x2(t_h_conv1)
            # self.state_w/2 * self.state_h/2 * 32
            # conv layer two
            t_h_conv2 = tf.nn.relu(self.conv2d(t_h_pool1, t_W_conv2) + t_b_conv2)
            # pooling layer one
            t_h_pool2 = self.max_pool_2x2(t_h_conv2)
            # self.state_w/4 * self.state_h/4 * 64
            # conv layer three
            # t_h_conv3 = tf.nn.relu(self.conv2d(t_h_pool2, t_W_conv3) + t_b_conv3)
            # self.state_w/4 * self.state_h/4 * 128
            t_h_conv2_flat = tf.reshape(t_h_pool2, [-1,int((self.state_w/4) * (self.state_h/4) * 64)])
            # hidden layer one
            t_h_layer_one = tf.nn.relu(tf.matmul(t_h_conv2_flat, t_W1) + t_b1)
            # dropout
            t_h_layer_one = tf.nn.dropout(t_h_layer_one, 1)
            # 防止过拟合
            # hidden layer two
            t_h_layer_two = tf.nn.relu(tf.matmul(t_h_layer_one, t_W2) + t_b2)
            # dropout
            t_h_layer_two = tf.nn.dropout(t_h_layer_two, 1)
            # the output of current_net
            target_Q_value = tf.matmul(t_h_layer_two, t_W3) + t_b3
            # dropout
            self.target_Q_value = tf.nn.dropout(target_Q_value, 1)
        # at last, solve the parameters replace problem
        # the parameters of current_net
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')
        # the parameters of target_net
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        # define the operation that replace the target_net's parameters by current_net's parameters
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def create_Q_network_defender(self):
        with tf.name_scope('inputs_defender'):
            # first, set the input of networks
            self.state_input_defender = tf.placeholder("float", [None, self.state_h, self.state_w, 15])
        # second, create the current_net
        with tf.variable_scope('current_net_defender'):
            # first, set the network's weights
            W_conv1 = self.weight_variable([5,5,15,32])
            b_conv1 = self.bias_variable([32])
            W_conv2 = self.weight_variable([5,5,32,64])
            b_conv2 = self.bias_variable([64])
            # W_conv3 = self.weight_variable([5,5,64,128])
            # b_conv3 = self.bias_variable([128])
            W1 = self.weight_variable([int((self.state_w/4) * (self.state_h/4) * 64), 512])
            b1 = self.bias_variable([512])
            W2 = self.weight_variable([512, 256])
            b2 = self.bias_variable([256])
            W3 = self.weight_variable([256, self.action_dim])
            b3 = self.bias_variable([self.action_dim])
            # second, set the layers
            # conv layer one
            h_conv1 = tf.nn.relu(self.conv2d(self.state_input_defender, W_conv1) + b_conv1)
            # self.state_w * self.state_h * 32
            # pooling layer one
            h_pool1 = self.max_pool_2x2(h_conv1)
            # self.state_w/2 * self.state_h/2 * 32
            # conv layer two
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            # pooling layer two
            h_pool2 = self.max_pool_2x2(h_conv2)
            # self.state_w/4 * self.state_h/4 * 64
            # conv layer three
            # h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
            # self.state_w/4 * self.state_h/4 * 128
            h_conv2_flat = tf.reshape(h_pool2, [-1,int((self.state_w/4) * (self.state_h/4) * 64)])
            # hidden layer one
            h_layer_one = tf.nn.relu(tf.matmul(h_conv2_flat, W1) + b1)
            # dropout
            h_layer_one = tf.nn.dropout(h_layer_one, 1)
            # hidden layer two
            h_layer_two = tf.nn.relu(tf.matmul(h_layer_one, W2) + b2)
            # dropout
            h_layer_two = tf.nn.dropout(h_layer_two, 1)
            # the output of current_net
            Q_value = tf.matmul(h_layer_two, W3) + b3
            # dropout
            self.Q_value_defender = tf.nn.dropout(Q_value, 1)
        # third, create the current_net
        with tf.variable_scope('target_net_defender'):
            # first, set the network's weights
            t_W_conv1 = self.weight_variable([5,5,15,32])
            t_b_conv1 = self.bias_variable([32])
            t_W_conv2 = self.weight_variable([5,5,32,64])
            t_b_conv2 = self.bias_variable([64])
            # t_W_conv3 = self.weight_variable([5,5,64,128])
            # t_b_conv3 = self.bias_variable([128])
            t_W1 = self.weight_variable([int((self.state_w/4) * (self.state_h/4) * 64), 512])
            t_b1 = self.bias_variable([512])
            t_W2 = self.weight_variable([512, 256])
            t_b2 = self.bias_variable([256])
            t_W3 = self.weight_variable([256, self.action_dim])
            t_b3 = self.bias_variable([self.action_dim])
            # second, set the layers
            # conv layer one
            t_h_conv1 = tf.nn.relu(self.conv2d(self.state_input_defender, t_W_conv1) + t_b_conv1)
            # self.state_w * self.state_h * 32
            # pooling layer one
            t_h_pool1 = self.max_pool_2x2(t_h_conv1)
            # self.state_w/2 * self.state_h/2 * 32
            # conv layer two
            t_h_conv2 = tf.nn.relu(self.conv2d(t_h_pool1, t_W_conv2) + t_b_conv2)
            # pooling layer one
            t_h_pool2 = self.max_pool_2x2(t_h_conv2)
            # self.state_w/4 * self.state_h/4 * 64
            # conv layer three
            # t_h_conv3 = tf.nn.relu(self.conv2d(t_h_pool2, t_W_conv3) + t_b_conv3)
            # self.state_w/4 * self.state_h/4 * 128
            t_h_conv2_flat = tf.reshape(t_h_pool2, [-1,int((self.state_w/4) * (self.state_h/4) * 64)])
            # hidden layer one
            t_h_layer_one = tf.nn.relu(tf.matmul(t_h_conv2_flat, t_W1) + t_b1)
            # dropout
            t_h_layer_one = tf.nn.dropout(t_h_layer_one, 1)
            # 防止过拟合
            # hidden layer two
            t_h_layer_two = tf.nn.relu(tf.matmul(t_h_layer_one, t_W2) + t_b2)
            # dropout
            t_h_layer_two = tf.nn.dropout(t_h_layer_two, 1)
            # the output of current_net
            target_Q_value = tf.matmul(t_h_layer_two, t_W3) + t_b3
            # dropout
            self.target_Q_value_defender = tf.nn.dropout(target_Q_value, 1)
        # at last, solve the parameters replace problem
        # the parameters of current_net
        e_params_defender = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net_defender')
        # the parameters of target_net
        t_params_defender = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net_defender')

        # define the operation that replace the target_net's parameters by current_net's parameters
        with tf.variable_scope('soft_replacement_defender'):
            self.target_replace_op_defender = [tf.assign(t, e) for t, e in zip(t_params_defender, e_params_defender)]

    # this the function that define the method to update the current_net's parameters
    def create_updating_method(self):
        # this the input action, use one hot presentation
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        # this the TD aim value
        self.y_input = tf.placeholder("float", [None])
        # this the action's Q_value
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices = 1)
        # 生成的Q_value实际上是一个action大小的list,action_input是一个one-hot向量,
        # 两者相乘实际上是取出了执行操作的Q值进行单独更新
        # this is the lost
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        # 均方差损失函数
        # drawing loss graph
        tf.summary.scalar('loss',self.cost)
        # loss graph save
        with tf.name_scope('train_loss'):
            # use the loss to optimize the network
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
            # learning_rate=0.0001


    def create_updating_method_defender(self):
        # this the input action, use one hot presentation
        self.action_input_defender = tf.placeholder("float", [None, self.action_dim])
        # this the TD aim value
        self.y_input_defender = tf.placeholder("float", [None])
        # this the action's Q_value
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value_defender, self.action_input_defender), reduction_indices = 1)
        # 生成的Q_value实际上是一个action大小的list,action_input是一个one-hot向量,
        # 两者相乘实际上是取出了执行操作的Q值进行单独更新
        # this is the lost
        self.cost_defender = tf.reduce_mean(tf.square(self.y_input_defender - Q_action))
        # 均方差损失函数
        # drawing loss graph
        tf.summary.scalar('loss_defender',self.cost_defender)
        # loss graph save
        with tf.name_scope('train_loss_defender'):
            # use the loss to optimize the network
            self.optimizer_defender = tf.train.AdamOptimizer(0.001).minimize(self.cost_defender)
            # learning_rate=0.0001

    # this is the function that use the network output the action
    def Choose_Action(self, state):
        # the output is a tensor, so the [0] is to get the output as a list
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        # use epsilon greedy to get the action
        if random.random() <= self.epsilon:
            # if lower than epsilon, give a random value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            # if bigger than epsilon, give the argmax value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)


    def Choose_Action_defender(self, state):
        # the output is a tensor, so the [0] is to get the output as a list
        Q_value = self.Q_value_defender.eval(feed_dict={
            self.state_input_defender: [state]
        })[0]
        # use epsilon greedy to get the action
        if random.random() <= self.epsilon:
            # if lower than epsilon, give a random value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            # if bigger than epsilon, give the argmax value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    # this the function that store the data in replay memory
    def Store_Data(self, state, action, reward, next_state, done):
        # generate a list with all 0,and set the action is 1
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        # store all the elements
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        # if the length of replay_buffer is bigger than REPLAY_SIZE
        # delete the left value, make the len is stable
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
            # update replay_buffer

    def Store_Data_defender(self, state, action, reward, next_state, done):
        # generate a list with all 0,and set the action is 1
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        # store all the elements
        self.replay_buffer_defender.append((state, one_hot_action, reward, next_state, done))
        # if the length of replay_buffer is bigger than REPLAY_SIZE
        # delete the left value, make the len is stable
        if len(self.replay_buffer_defender) > REPLAY_SIZE:
            self.replay_buffer_defender.popleft()
            # update replay_buffer

    # train the network, update the parameters of Q_value
    def Train_Network(self, BATCH_SIZE, num_step):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        #print("minibatch: ",len(minibatch))
        # 从记忆库中采样BATCH_SIZE
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate TD aim value
        y_batch = []
        # give the next_state_batch flow to target_Q_value and caculate the next state's Q_value
        Q_value_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})
        # caculate the TD aim value by the formulate
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            # see if the station is the final station
            if done:
                y_batch.append(reward_batch[i])
            else:
                # the Q value caculate use the max directly
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        # step 3: update the network
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            # y即为更新后的Q值,与Q_action构成损失函数更新网络
            self.action_input: action_batch,
            self.state_input: state_batch
        })
        #if num_step % 100 == 0:
            # save loss graph
            # print(len(state_batch))
            # result = self.sess.run(self.merged,feed_dict={
            # self.y_input: y_batch,
            # self.action_input: action_batch,
            # self.state_input: state_batch
            # })
            # # 把merged的数据放进writer中才能画图
            # self.writer.add_summary(result, num_step)


    def Train_Network_defender(self, BATCH_SIZE, num_step):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer_defender, BATCH_SIZE)
        #print("minibatch: ",len(minibatch))
        # 从记忆库中采样BATCH_SIZE
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate TD aim value
        y_batch = []
        # give the next_state_batch flow to target_Q_value and caculate the next state's Q_value
        Q_value_batch = self.target_Q_value_defender.eval(feed_dict={self.state_input_defender: next_state_batch})
        # caculate the TD aim value by the formulate
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            # see if the station is the final station
            if done:
                y_batch.append(reward_batch[i])
            else:
                # the Q value caculate use the max directly
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        # step 3: update the network
        self.optimizer_defender.run(feed_dict={
            self.y_input_defender: y_batch,
            # y即为更新后的Q值,与Q_action构成损失函数更新网络
            self.action_input_defender: action_batch,
            self.state_input_defender: state_batch
        })
        #if num_step % 100 == 0:
            # # save loss graph
            # result = self.sess.run(self.merged_defender,feed_dict={
            # self.y_input_defender: y_batch,
            # self.action_input_defender: action_batch,
            # self.state_input_defender: state_batch
            # })
            # # 把merged的数据放进writer中才能画图
            # self.writer_defender.add_summary(result, num_step)

    def Update_Target_Network(self):
        # update target Q netowrk
        self.sess.run(self.target_replace_op)


    def Update_Target_Network_defender(self):
        # update target Q netowrk
        self.sess.run(self.target_replace_op_defender)

    # use for test
    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def action_defender(self, state):
        return np.argmax(self.Q_value_defender.eval(feed_dict={
            self.state_input_defender: [state]
        })[0])
    
    def save_model(self):
        self.save_path = self.saver.save(self.sess, self.model_path)
        print("Save to path:", self.save_path)
        #print("=========LAST TIME All Variables==========")
        #print_tensors_in_checkpoint_file(self.model_path, tensor_name=None, all_tensors=True, all_tensor_names=True)


    def save_model_defender(self):
        self.save_path_defender = self.saver_defender.save(self.sess, self.model_path_defender)
        print("Save to path:", self.save_path_defender)
        