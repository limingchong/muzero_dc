# -*- codeing = utf-8 -*-
# @Time:2021/9/25 9:39
# @Author: ZhangHanwen
# @File:a_inv_7.py
# @Software: PyCharm
from tensorflow import *
from keras.preprocessing.image import ImageDataGenerator
import keras
# from keras.models import Sequential
from keras import layers
# from keras.metrics import categorical_accuracy

import tensorflow.keras.optimizers
# import numpy as np
# import os
# import shutil
# from tensorflow.keras.utils import to_categorical
# wj_dir='C:\python_project/set/zwjset'
sets = ImageDataGenerator(
        rescale = 1./255
        # shear_range = 0.2,
        # zoom_range = 0.2,
        # horizontal_flip=True
)
        #归一化验证集

sets_wjhw='C:\python_project\set_dct'
# 制作归一化模型
a_inv_face_model = sets.flow_from_directory(sets_wjhw,
                                            batch_size=1,
                                            # class_mode='binary',
                                            # class_mode='binary',
                                            class_mode='binary',
                                            # class_mode='categorical',
                                            # color_mode='grayscale'
                                            color_mode='rgb'
                                            )
print(a_inv_face_model.labels)
print(a_inv_face_model)
# sdkjfbkjsdbvkjdsb
# wj_model = wj_gen.flow_from_directory(wj_gen,
#                                       target_size=(128,128),
#                                       batch_size=1,
#                                       class_mode='categorical'
# )                                            class_mode='binary',

# print(train_datagenerator)
# print(test_datagenerator)

# a_inv_face_model.labels = np.array([0 if label.endswith('zhwset') else 1 for label in sets_wjhw])
model=keras.Sequential()
# 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
model.add(layers.Conv2D(32,(1, 1),padding='same',input_shape=a_inv_face_model.image_shape,activation='relu'))  # 1 2维卷积层
# model.add(layers.Activation('relu'))  # 2 激活函数层
model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))  # 7  2维卷积层
model.add(layers.MaxPooling2D((3,3),strides=2)) # 5 池化层
# model.add(layers.Conv2D(32,(5,5),padding='same',activation='relu'))  # 7  2维卷积层
# model.add(layers.Activation('relu'))  # 2 激活函数层
# model.add(layers.MaxPooling2D()) # 5 池化层
model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))  # 7  2维卷积层
# model.add(layers.MaxPooling2D()) # 5 池化层
model.add(layers.Conv2D(64,(5,5),padding='same',activation='relu'))  # 7  2维卷积层
model.add(layers.Conv2D(32,(9,9),padding='same',activation='relu'))  # 7  2维卷积层
# model.add(layers.MaxPooling2D()) # 5 池化层
# model.add(layers.Conv2D(128,(5,5),padding='same',activation='relu'))  # 7  2维卷积层

# model.add(layers.MaxPooling2D()) # 5 池化层
# model.add(layers.Conv2D(256,(5,5),padding='same',activation='relu'))  # 7  2维卷积层
# model.add(layers.Conv2D(256,(5,5),padding='same',activation='relu'))  # 7  2维卷积层

# model.add(layers.Activation('relu'))  # 2 激活函数层
model.add(layers.Flatten())  # 13 Flatten层

model.add(layers.Dense(256))  # 14 Dense层,又被称作全连接层
model.add(layers.Activation('relu'))  # 15 激活函数层
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='relu'))  # 17 Dense层
model.summary()

sgd=tensorflow.keras.optimizers.SGD(lr=0.0001, decay=1e-4, momentum=0.5, nesterov=True)

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=[categorical_accuracy])
model.fit_generator(a_inv_face_model,epochs=10,steps_per_epoch=100)
model.save('./model.h5')




#
# # -*- codeing = utf-8 -*-
# # @Time:2021/9/25 9:39
# # @Author: ZhangHanwen
# # @File:a_inv_7.py
# # @Software: PyCharm
# import tensorflow.keras as keras
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.metrics import categorical_accuracy
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import tensorflow.keras.optimizers
# import numpy as np
# import os
# import shutil
# from tensorflow.keras.utils import to_categorical
# # wj_dir='../input/training'
# sets = ImageDataGenerator(
#         rescale = 1./255,
#         shear_range = 0.2,
#         zoom_range = 0.2,
#         horizontal_flip=True)
#         #归一化验证集
#
# sets_wjhw='C:\python_project/real-time-emotion-detection-master/fer2013_rename/training'
# # 制作归一化模型
# a_inv_face_model = sets.flow_from_directory(sets_wjhw,
#                                             batch_size=10,
#                                             # class_mode='binary',
#                                             # class_mode='binary',
#                                             class_mode='categorical',
#                                             color_mode='grayscale'
#                                             )
# print(a_inv_face_model.labels)
# print(a_inv_face_model)
# # sdkjfbkjsdbvkjdsb
# # wj_model = wj_gen.flow_from_directory(wj_gen,
# #                                       target_size=(128,128),
# #                                       batch_size=1,
# #                                       class_mode='categorical'
# # )                                            class_mode='binary',
#
# # print(train_datagenerator)
# # print(test_datagenerator)
#
# # a_inv_face_model.labels = np.array([0 if label.endswith('zhwset') else 1 for label in sets_wjhw])
# model=keras.Sequential()
# # 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
# model.add(layers.Conv2D(32,(5, 5),padding='same',input_shape=a_inv_face_model.image_shape,activation='relu'))  # 1 2维卷积层
# # model.add(layers.Activation('relu'))  # 2 激活函数层
# # model.add(layers.MaxPooling2D((3,3),strides=2)) # 5 池化层
# # model.add(layers.Conv2D(32,(5,5),padding='same',activation='relu'))  # 7  2维卷积层
# # model.add(layers.Activation('relu'))  # 2 激活函数层
# model.add(layers.MaxPooling2D()) # 5 池化层
# model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))  # 7  2维卷积层
# # model.add(layers.MaxPooling2D()) # 5 池化层
# model.add(layers.Conv2D(64,(5,5),padding='same',activation='relu'))  # 7  2维卷积层
# # model.add(layers.Conv2D(64,(5,5),padding='same',activation='relu'))  # 7  2维卷积层
# # model.add(layers.MaxPooling2D()) # 5 池化层
# model.add(layers.Conv2D(128,(5,5),padding='same',activation='relu'))  # 7  2维卷积层
#
# # model.add(layers.MaxPooling2D()) # 5 池化层
# model.add(layers.Conv2D(256,(5,5),padding='same',activation='relu'))  # 7  2维卷积层
# model.add(layers.Conv2D(256,(5,5),padding='same',activation='relu'))  # 7  2维卷积层
#
# # model.add(layers.Activation('relu'))  # 2 激活函数层
# model.add(layers.Flatten())  # 13 Flatten层
#
# model.add(layers.Dense(512))  # 14 Dense层,又被称作全连接层
# model.add(layers.Activation('relu'))  # 15 激活函数层
# # model.add(layers.Dropout(0.5))
# model.add(layers.Dense(7,activation='softmax'))  # 17 Dense层
# model.summary()
#
# sgd=tensorflow.keras.optimizers.SGD(lr=0.0001, decay=1e-4, momentum=0.5, nesterov=True)
#
# # model.compile(loss='categorical_crossentropy',
# #               optimizer='adam',
# #               metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy',
#               optimizer='adagrad',
#               metrics=['accuracy'])
# # model.compile(loss='binary_crossentropy',
# #               optimizer='adam',
# #               metrics=[categorical_accuracy])
# model.fit_generator(a_inv_face_model,epochs=10,steps_per_epoch=100)
# model.save('./emotion.h5')
