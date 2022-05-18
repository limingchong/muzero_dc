from DQN_tensorflow_gpu import DQN
import tensorflow.compat.v1 as tf


config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)