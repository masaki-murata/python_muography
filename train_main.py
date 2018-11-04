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
from keras.layers import Input, Dense, Conv3D, Flatten
from keras.models import Model
from keras.optimizers import Adam
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
def deform_time(time=1, # 噴火までの時間を時間単位で
                prediction_hour=24, # 噴火を予期したい時間を時間単位で
                ):
    if time > prediction_hour:
        arg_tanh = (time-prediction_hour) / prediction_hour
        time = prediction_hour + prediction_hour*math.tanh(arg_tanh)
    return time

# CNN モデルを作る関数
def make_model(input_shape,
               ):
    input_img = Input(shape=input_shape)
    x = Conv3D(filters=2, kernel_size=(2,2,2), padding="valid", activation="relu")(input_img)
    output_conv = Conv3D(filters=2, kernel_size=(2,2,2), padding="valid", activation="relu")(x)
    
    x = Flatten()(output_conv)
    x = Dense(256, activation="relu")(x)
    output = Dense(1, activation="relu")(x)
    
    model = Model(input_img, output)
    opt_generator = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=opt_generator)
    
    model.summary()

# データの重複がないように train, validation, test に分ける
def devide_data(#end_of_observations=[],
                #path_to_observation_hour_csv = "../data/observationhour%03d.csv",
                days_period=10,
                observation_hour=6, # period_observation 時間の観測データを使う
                ratio = [0.7, 0.15, 0.15],
                ):
    path_to_csv =  "../data/observation_dp%03d.csv" % (days_period)

    df_observation = pd.read_csv(path_to_csv)
    end_of_observations = read_data.remove_time_deficit(observation_hour)
    time_step_observation = observation_hour*6 # １０分単位に変換
    train_num = int(len(end_of_observations)*ratio[0])
    validation_num = int(len(end_of_observations)*ratio[1])
    
    # ３つに分割
    eoo_train = end_of_observations[:train_num]
    eoo_validation = end_of_observations[train_num+time_step_observation:train_num+validation_num]
    eoo_test = end_of_observations[train_num+validation_num+time_step_observation:]
    
    return eoo_train, eoo_validation, eoo_test
    
#    df_train = df_observation[df_observation["end of observation"].isin(eoo_train)]
#    df_train = df_observation[:train_num]
#    df_validation = df_observation[train_num+time_step_observation:train_num+validation_num]
#    df_test = df_observation[train_num+validation_num+time_step_observation:]
#    
#    return df_train, df_validation, df_test

# validation と test ではデータの重複がないように抽出
def make_validation_test(df,
                         observation_hour=6, # period_observation 時間の観測データを使う
                         prediction_hour=24, # prediction_hour 時間までの予測ができるように
                         sample_size_half=50,
                         ):
    time_step_observation = observation_hour*6 # １０分単位に変換
    df_short = df[df["time to eruption"] <= prediction_hour]
    df_long = df[df["time to eruption"] > prediction_hour]
    print(len(df_short), len(df_long))
    # eoo = end of observation
    eoo_short = df_short["end of observation"].values
    eoo_long = df_long["end of observation"].values
    eoo_short = list(map(lambda x: read_data.str_to_datetime(x,slash_dash="dash"), eoo_short))
    eoo_long = list(map(lambda x: read_data.str_to_datetime(x,slash_dash="dash"), eoo_long))
#    print(len(eoo_short), len(eoo_long))
#    print(type(eoo_short))
    time_delta = datetime.timedelta(0, observation_hour*3600)
#    print(time_delta)
    eoo_sample = []
#    eoo_sample_short = []*sample_size_half
    for count in range(sample_size_half*2):
        if count % 2==0:
            eoo = random.choice(eoo_short)
        else:
            eoo = random.choice(eoo_long)
        eoo_sample.append(eoo)
        eoo_short = [x_short for x_short in eoo_short if abs(x_short-eoo)>=time_delta]
        eoo_long = [x_long for x_long in eoo_long if abs(x_long-eoo)>=time_delta]
#        print(len(eoo_short))
        
        count += 1
        assert len(eoo_short)+len(eoo_long) > len(df) - (time_step_observation*2+1)*count - 1, \
        "{0},{1},{2},{3}".format(len(eoo_short), len(eoo_long), len(df), count)
    eoo_sample = list(map(lambda x: read_data.datetime_to_str(x), eoo_sample))
    
    df = df[df["end of observation"].isin(eoo_sample)]
    
    return df
#    time_to_eruptions = 

def df_to_data(df,
               prediction_hour=24,
               observtion_hour=6,
               input_shape=(29,29,1),
               ):
    data = np.array(df.loc[:,"pixel001":"pixel841"])
    data = data.reshape((len(data),)+input_shape)
    label = df["time to eruption"].values
    label = np.array([deform_time(time, prediction_hour) for time in label])
    label = label.reshape((len(label),1))

    return data, label


def batch_iter(df_train,
               prediction_hour=24,
               input_shape=(29,29,1),
#               train_data,
#               train_label,
               batch_size=32,
               ):
    data_num = len(df_train)
    steps_per_epoch = int( (data_num - 1) / batch_size ) + 1
    def data_generator():
        while True:
            for batch_num in range(steps_per_epoch):
                if batch_num==0:
#                    p = np.random.permutation(len(train_data))
#                    data_shuffled = train_data[p]
#                    label_shuffled = train_label[p]
                    df_shuffle = df_train.sample(frac=1)
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_num)
                df_epoch = df_shuffle[start_index:end_index]
                data, labels = df_to_data(df=df_epoch, 
                                          prediction_hour=prediction_hour, 
                                          input_shape=input_shape)
                
                yield data, labels
    
    return data_generator(), steps_per_epoch


def train(input_shape=(29,29,1),
          ratio=[0.7,0.15,0.15],
          days_period=30,
          observation_hour=6,
          prediction_hour=24,
          val_sample_size_half=50,
          test_sample_size_half=50,
          epochs=10,
          batch_size=128,
          nb_gpus=1,
          ):
    
    # load data
    eoo_train, eoo_validation, eoo_test = devide_data(days_period=days_period,
                                                      observation_hour=observation_hour,
                                                      ratio=ratio,
                                                      )
    
#    df_train, df_validation, df_test = devide_data(days_period=days_period,
#                                                   observation_hour=observation_hour,
#                                                   ratio=ratio,
#                                                   )
    
    df_validation=make_validation_test(df=df_validation,
                                       sample_size_half=val_sample_size_half,
                                       observation_hour=observation_hour)
    df_test=make_validation_test(df=df_test, 
                                 sample_size_half=test_sample_size_half,
                                 observation_hour=observation_hour)
    print("data, label")
    
#    train_data, train_label = df_to_data(df=df_train, prediction_hour=prediction_hour)
    val_data, val_label = df_to_data(df=df_validation, prediction_hour=prediction_hour)
    test_data, test_label = df_to_data(df=df_test, prediction_hour=prediction_hour)

    # load model
    model = make_model(input_shape=input_shape)
    if int(nb_gpus) > 1:
        model_multiple_gpu = multi_gpu_model(model, gpus=nb_gpus)
    else:
        model_multiple_gpu = model
        
    # train ようのデータジェネレータを作成
    train_gen , steps_per_epoch= batch_iter(df_train,
                                            batch_size=batch_size
                                            )
    
    # train
    model_multiple_gpu.fit_generator(train_gen,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=epochs,
                                     validation_data=(val_data,val_label)
                                     )
    return model

def evaluate_test(df_test,
                  prediction_hour=24,
                  path_to_model="",
                  path_to_predicted="",
                  model="empty",
                  batch_size=128,
                  ):
    # df からデータ取り出し
    test_data, test_label = df_to_data(df=df_test, prediction_hour=prediction_hour)

    if model=="empty":
        model = keras.model.load_model(path_to_model)
        
    predicted = model.predict(test_data, batch_size=batch_size)
    
    # dataframe として推定値と実際の値をリスト化
    df_predicted = pd.DataFrame(predicted, columns=["predicted time to eruption"])

    df_truth = df_test.loc[:, ["end of observation", "time to eruption"]]
    df_truth = df_truth.rename(columns={"time to eruption":"actual time to eruption"})
    
    df_predicted = pd.concat([df_truth,df_predicted], axis=1)    
    df_predicted.to_csv(path_to_predicted, index=None)

    
def main():     
#    df_train, df_validation, df_test = devide_data(observation_hour=6)
#    df_validation=make_validation_test(df=df_validation, observation_hour=6)
#    print(len(df_validation))
#    print(df_validation.columns)
#    val_data, val_label = df_to_data(df_validation, prediction_hour=24)
#    print(val_data.shape)
#    make_model(input_shape=(29,29,144,1))  
    train(input_shape=(29,29,1),
          ratio=[0.5,0.25,0.25],
          days_period=30,
          observation_hour=6,
          prediction_hour=24,
          val_sample_size_half=50,
          test_sample_size_half=50,
          epochs=10,
          batch_size=128,
          nb_gpus=1,
          )
    
if __name__ == '__main__':
    main()

