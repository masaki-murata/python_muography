#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:24:58 2018

@author: muratamasaki
"""
import numpy as np
import pandas as pd
import random, datetime, math, os
import keras
from keras.layers import Input, Dense, Conv3D, Flatten, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
if os.name=='posix':
    from keras.utils.training_utils import multi_gpu_model

import read_data

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

if os.name=='posix':
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="1", # specify GPU number
            allow_growth=True
        )
    )
    
    set_session(tf.Session(config=config))

# 噴火までの時間を圧縮する関数
def deform_time(time, # 噴火までの時間を時間単位で
                prediction_hour, # 噴火を予期したい時間を時間単位で
                ):
    if time > prediction_hour:
        arg_tanh = (time-prediction_hour) / prediction_hour
        time = prediction_hour + prediction_hour*math.tanh(arg_tanh)
    return time

# CNN モデルを作る関数
def make_model(input_shape,
               ):
    input_img = Input(shape=input_shape)
    x = Conv3D(filters=8, kernel_size=(3,3,3), padding="valid", activation="relu")(input_img)
    x = Conv3D(filters=8, kernel_size=(3,3,3), strides=(2,2,2), padding="valid", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=32, kernel_size=(3,3,3), padding="valid", activation="relu")(x)
    x = Conv3D(filters=32, kernel_size=(3,3,3), strides=(2,2,2), padding="valid", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=128, kernel_size=(3,3,3), padding="valid", activation="relu")(x)
    x = Conv3D(filters=128, kernel_size=(3,3,3), padding="valid", activation="relu")(x)
#    x = Conv3D(filters=128, kernel_size=(3,3,3), strides=(2,2,2), padding="valid", activation="relu")(x)
#    x = BatchNormalization()(x)
    
#    output_conv = Conv3D(filters=2, kernel_size=(2,2,2), padding="valid", activation="relu")(x)
    
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(1, activation="relu")(x)
    
    model = Model(input_img, output)
    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=opt_generator, metrics=['mae'])
    
    model.summary()
    
    return model

# データの重複がないように train, validation, test に分ける
def devide_data(#end_of_observations=[],
                #path_to_observation_hour_csv = "../data/observationhour%03d.csv",
                days_period=10,
                observation_hour=6, # period_observation 時間の観測データを使う
                ratio = [0.7, 0.15, 0.15],
                ):
    path_to_csv =  "../data/observation_daysperiod%03d.csv" % (days_period)
    df = pd.read_csv(path_to_csv)
#    df_observation = pd.read_csv(path_to_csv)
    end_of_observations = read_data.remove_time_deficit(df,observation_hour=observation_hour)
    time_step_observation = observation_hour*6 # １０分単位に変換
    train_num = int(len(end_of_observations)*ratio[0])
    validation_num = int(len(end_of_observations)*ratio[1])
    
    # ３つに分割
    eoos_train = end_of_observations[:train_num]
    eoos_validation = end_of_observations[train_num+time_step_observation:train_num+validation_num]
    eoos_test = end_of_observations[train_num+validation_num+time_step_observation:]
    
    return eoos_train, eoos_validation, eoos_test
    
#    df_train = df_observation[df_observation["end of observation"].isin(eoo_train)]
#    df_train = df_observation[:train_num]
#    df_validation = df_observation[train_num+time_step_observation:train_num+validation_num]
#    df_test = df_observation[train_num+validation_num+time_step_observation:]
#    
#    return df_train, df_validation, df_test

# validation と test では画像の重複がないように eoo を抽出
def make_validation_test(df,
                         eoos=[],
                         observation_hour=6, # observation_hour 時間の観測データを使う
                         prediction_hour=24, # prediction_hour 時間までの予測ができるように
                         sample_size_half=50,
                         ):
    time_step_observation = observation_hour*6 # １０分単位に変換
    df = df[df["end of observation"].isin(eoos)]
    df_short = df[df["time to eruption"] <= prediction_hour]
    df_long = df[df["time to eruption"] > prediction_hour]
    print(len(df_short), len(df_long))
    # eoo = end of observation
    eoos_short = df_short["end of observation"].values
    eoos_long = df_long["end of observation"].values
    eoos_short = list(map(lambda x: read_data.str_to_datetime(x,slash_dash="dash"), eoos_short))
    eoos_long = list(map(lambda x: read_data.str_to_datetime(x,slash_dash="dash"), eoos_long))
#    print(len(eoo_short), len(eoo_long))
#    print(type(eoo_short))
    time_delta = datetime.timedelta(0, observation_hour*3600)
#    print(time_delta)
    eoos_sample = []
#    eoo_sample_short = []*sample_size_half
    for count in range(sample_size_half*2):
        if count % 2==0:
            eoo = random.choice(eoos_short)
        else:
            eoo = random.choice(eoos_long)
        eoos_sample.append(eoo)
        eoos_short = [x_short for x_short in eoos_short if abs(x_short-eoo)>=time_delta]
        eoos_long = [x_long for x_long in eoos_long if abs(x_long-eoo)>=time_delta]
#        print(len(eoo_short))
        
        count += 1
        assert len(eoos_short)+len(eoos_long) > len(df) - (time_step_observation*2+1)*count - 1, \
        "{0},{1},{2},{3}".format(len(eoos_short), len(eoos_long), len(df), count)
    eoo_sample = list(map(lambda x: read_data.datetime_to_str(x), eoos_sample))
    
    return eoo_sample
    
#    df = df[df["end of observation"].isin(eoo_sample)]
#    
#    return df
#    time_to_eruptions = 

# 観測終了時間が end of obsevation となる画像を作成
def eoo_to_data(df, 
                eoo=[],
#                prediction_hour=24,
                observation_hour=6,
                image_shape=(29,29,1),
                ):
    movie_shape = (observation_hour*6-1,) + image_shape
    ten_minutes = datetime.timedelta(minutes=10)
    eoo_datetime = read_data.str_to_datetime(eoo, slash_dash="dash")
    eoos_datetime = [(eoo_datetime - ten_minutes*ts) for ts in range(observation_hour*6-1)]
    eoos_str = [read_data.datetime_to_str(eoo_datetime) for eoo_datetime in eoos_datetime]
#    time_delta = datetime.timedelta(minutes=(observation_hour*60-10))
#    initial_time = read_data.str_to_datetime(eoo, slash_dash="dash") - time_delta
    
#    eoo_list = [time for x in range]
#    initial_time = read_data.str_to_datetime(initial_time)
    df = df[df["end of observation"].isin(eoos_str)]
#    df = df.loc[initial_time:eoo]
    
    image = np.array(df.loc[:,"pixel001":"pixel841"])
    image = image.reshape(movie_shape) # この reshape が想定通り動くかはチェックする必要あり
    label = df[-1:]["time to eruption"].values[0]
#    label = np.array([deform_time(time, prediction_hour) for time in label])
#    label = label.reshape((len(label),1))
    
    return image, label

# eoo_to_data の複数版
def eoos_to_data(df,
                 eoos=[],
#               prediction_hour=24,
                 observation_hour=6,
                 image_shape=(29,29,1),
                 ):
    movie_shape = (observation_hour*6-1,) + image_shape
    input_shape = (len(eoos),) + movie_shape
    data = np.zeros(input_shape)
    labels = np.zeros((len(eoos), 1))
    count = 0
    for eoo in eoos:
        data[count], labels[count] = eoo_to_data(df, eoo, observation_hour=observation_hour)
        count += 1
#    df = df[df["end of observation"].isin(eoo)]
#    data = np.array(df.loc[:,"pixel001":"pixel841"])
#    data = data.reshape((len(data),)+input_shape)
#    label = df["time to eruption"].values
#    label = np.array([deform_time(time, prediction_hour) for time in label])
#    label = label.reshape((len(label),1))

    return data, labels

## training データをとにかく全部つくる関数
#def make_train_data(df,
#                    eoos_train):

# バッチごとに numpy からトレーニング用のデータとラベルを作る関数
def batch_iter_np(data, labels,
                  prediction_hour,
                  batch_size,
                  ):
    batch_size_half = batch_size // 2
    data_short = data[np.where(labels<=prediction_hour)[0]]
    labels_short = labels[np.where(labels<=prediction_hour)[0]]
    data_long = data[np.where(labels>prediction_hour)[0]]
    labels_long = labels[np.where(labels>prediction_hour)[0]]

    data_num_half = min(len(labels_short), len(labels_long))
    steps_per_epoch = int( (data_num_half - 1) / batch_size_half ) + 1

    def data_generator():
        while True:
            for batch_num in range(steps_per_epoch):
                if batch_num==0:
                    short_indices = np.random.randint( len( data_short ), size=data_num_half )
                    data_short_shuffled = data_short[short_indices]
                    labels_short_shuffled = labels_short[short_indices]
                    long_indices = np.random.randint( len( data_long ), size=data_num_half )
                    data_long_shuffled = data_long[long_indices]
                    labels_long_shuffled = labels_long[long_indices]

                start_index = batch_num * batch_size_half
                end_index = min((batch_num + 1) * batch_size_half, data_num_half)
                batch_data = np.vstack((data_short_shuffled[start_index:end_index], data_long_shuffled[start_index:end_index]))
                batch_labels = np.vstack((labels_short_shuffled[start_index:end_index], labels_long_shuffled[start_index:end_index]))
                batch_indices = np.random.randint( len( batch_labels ), size=len(batch_labels) )
                
                yield batch_data[batch_indices], batch_labels[batch_indices]

    return steps_per_epoch, data_generator()
    
    
# バッチごとに df からトレーニング用のデータとラベルを作る関数
def batch_iter(df,
               eoos_train=[],
               prediction_hour=24,
               observation_hour=6,
               image_shape=(29,29,1),
#               train_data,
#               train_label,
               batch_size=128,
               ):
    
    data_num = len(eoos_train)
    steps_per_epoch = int( (data_num - 1) / batch_size ) + 1
#    print("steps_per_epoch={0}".format(steps_per_epoch))
    def data_generator():
        while True:
            for batch_num in range(steps_per_epoch):
                if batch_num==0:
#                    print("shuffle")
                    random.shuffle(eoos_train)
#                    p = np.random.permutation(len(train_data))
#                    data_shuffled = train_data[p]
#                    label_shuffled = train_label[p]
#                    df_shuffle = df.sample(frac=1)
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_num)
                eoos_epoch = eoos_train[start_index:end_index]
#                df_epoch = df_shuffle[start_index:end_index]
                batch_data, batch_labels = eoos_to_data(df=df, 
                                                        eoos=eoos_epoch,
                                                        observation_hour=observation_hour,
                                                        image_shape=image_shape,
                                                        )
#                print("data.shape={0}, labels.shape={1}".format(data.shape, labels.shape))
                yield batch_data, batch_labels
    
    return steps_per_epoch, data_generator()


def train(image_shape=(29,29,1),
          ratio=[0.7,0.15,0.15],
          days_period=30,
          observation_hour=6,
          prediction_hour=24,
          val_sample_size_half=50,
          test_sample_size_half=50,
          epochs=10,
          batch_size=128,
          if_generator_df=True,
          nb_gpus=1,
          ):
    
    path_to_observation = "../data/observation_daysperiod%03d.csv" % (days_period)
    if os.path.exists(path_to_observation):
        df = pd.read_csv(path_to_observation)
    else:
        df = read_data.remove_no_eruption_period(days_period=days_period)
    
    # load data
    eoos_train, eoos_validation, eoos_test = devide_data(days_period=days_period,
                                                         observation_hour=observation_hour,
                                                         ratio=ratio,
                                                         )

    # 長時間部分を圧縮
    print("prediction_hour = ", prediction_hour)
    df["time to eruption"] = df["time to eruption"].map(lambda time: deform_time(time,prediction_hour))
    print(max(df["time to eruption"].values))
#    eoos_train = [deform_time(eoo, prediction_hour) for eoo in eoos_train]
#    eoos_validation = [deform_time(eoo,prediction_hour) for eoo in eoos_validation]
#    eoos_test = [deform_time(eoo,prediction_hour) for eoo in eoos_test]
#    print(max(eoos_validation))
    # sampling for validation and test
    eoos_validation=make_validation_test(df,
                                         eoos=eoos_validation,
                                         sample_size_half=val_sample_size_half,
                                         observation_hour=observation_hour,
                                         prediction_hour=prediction_hour)
    eoos_test=make_validation_test(df, 
                                   eoos=eoos_test,
                                   sample_size_half=test_sample_size_half,
                                   observation_hour=observation_hour,
                                   prediction_hour=prediction_hour)

    
#    df_train, df_validation, df_test = devide_data(days_period=days_period,
#                                                   observation_hour=observation_hour,
#                                                   ratio=ratio,
#                                                   )
    
#    df_validation=make_validation_test(df=df_validation,
#                                       sample_size_half=val_sample_size_half,
#                                       observation_hour=observation_hour)
#    df_test=make_validation_test(df=df_test, 
#                                 sample_size_half=test_sample_size_half,
#                                 observation_hour=observation_hour)
    
#    train_data, train_label = df_to_data(df=df_train, prediction_hour=prediction_hour)
    val_data, val_label = eoos_to_data(df=df, eoos=eoos_validation, observation_hour=observation_hour)
    test_data, test_label = eoos_to_data(df=df, eoos=eoos_test, observation_hour=observation_hour)

    # load model
    movie_shape = (observation_hour*6-1,) + image_shape
    model = make_model(input_shape=movie_shape)
    if int(nb_gpus) > 1:
        model_multiple_gpu = multi_gpu_model(model, gpus=nb_gpus)
    else:
        model_multiple_gpu = model
        
    # train 用のデータジェネレータを作成
    if if_generator_df:
        steps_per_epoch, train_gen = batch_iter(df,
                                                eoos_train=eoos_train,
                                                prediction_hour=prediction_hour,
                                                observation_hour=observation_hour,
                                                image_shape=image_shape,
                                                batch_size=batch_size,
                                                )
    else:
        train_data, train_label = eoos_to_data(df,
                                               eoos=eoos_train,
                                               observation_hour=observation_hour,
                                               image_shape=image_shape,
                                               )
        steps_per_epoch, train_gen = batch_iter_np(train_data, train_label,
                                                   prediction_hour=prediction_hour,
                                                   batch_size=batch_size,
                                                   )
        print(np.average(train_label), np.average(val_label))

        
    print("start train")
    # train
    print(val_label.shape)
    model_multiple_gpu.fit_generator(generator=train_gen,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=epochs,
                                     validation_data=(val_data,val_label),
                                     )
#    if if_generator_df:
#        model_multiple_gpu.fit_generator(generator=train_gen,
#                                         steps_per_epoch=steps_per_epoch,
#                                         epochs=epochs,
#                                         validation_data=(val_data,val_label),
#                                         )
#    else:
#        model_multiple_gpu.fit(train_data, train_label,
#                               batch_size=batch_size,
#                               epochs=epochs,
#                               validation_data=(val_data,val_label),
#                               )
    return model

def evaluate_test(df,
                  eoos_test=[],
                  observation_hour=6,
                  prediction_hour=24,
                  path_to_model="",
                  path_to_predicted="",
                  model="empty",
                  batch_size=128,
                  ):
    # df からデータ取り出し
    test_data, test_label = eoos_to_data(df=df, eoos=eoos_test, observation_hour=observation_hour)

    if model=="empty":
        model = keras.model.load_model(path_to_model)
        
    predicted = model.predict(test_data, batch_size=batch_size)
    
    # dataframe として推定値と実際の値をリスト化
    df_predicted = pd.DataFrame(predicted, columns=["predicted time to eruption"])

    df = df[df["end of observation"].isin(eoos_test)]
    df_truth = df.loc[:, ["end of observation", "time to eruption"]]
    df_truth = df_truth.rename(columns={"time to eruption":"actual time to eruption"})
    
    df_predicted = pd.concat([df_truth,df_predicted], axis=1)    
    df_predicted.to_csv(path_to_predicted, index=None)

    
def main():
    image_shape=(29,29,1)
    days_period=10
    observation_hour=6
    prediction_hour=6
    val_sample_size_half=50
    test_sample_size_half=50
    ratio=[0.6, 0.2, 0.2]
    epochs=100
    batch_size=256
    if_generator_df=False
    nb_gpus=1
    
#    eoos_train, eoos_validation, eoos_test = devide_data(days_period=30, observation_hour=6)
##    print(eoos_train[0])
#    path_to_observation = "../data/observation_daysperiod%03d.csv" % (days_period)
#    df = pd.read_csv(path_to_observation)
##    image, label = eoo_to_data(df, eoos_train[0])
#    data, labels = df_to_data(df, eoos_train, observation_hour=observation_hour)
#    print(type(data))
#    print(type(labels))

    train(image_shape=image_shape,
          ratio=ratio,
          days_period=days_period,
          observation_hour=observation_hour,
          prediction_hour=prediction_hour,
          val_sample_size_half=val_sample_size_half,
          test_sample_size_half=test_sample_size_half,
          epochs=epochs,
          batch_size=batch_size,
          if_generator_df=if_generator_df,
          nb_gpus=nb_gpus,
          )
    
if __name__ == '__main__':
    main()

