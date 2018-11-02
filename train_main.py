#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:24:58 2018

@author: muratamasaki
"""

import pandas as pd
import random, datetime
from keras.layers import Input, Dense, Conv3D, Flatten
from keras.models import Model
from keras.optimizers import Adam

import read_data

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

def devide_data(path_to_observation_time_step_csv = "../data/observation_timestep%03d.csv",
                time_step=6, # time_step*10 分間の観測データを使う
                ratio = [0.6, 0.2, 0.2],
                ):
    df_observation = pd.read_csv(path_to_observation_time_step_csv % time_step)
    train_num = int(len(df_observation)*ratio[0])
    validation_num = int(len(df_observation)*ratio[1])
#    test_num = len(df_observation) - train_num - validation_num
    
    # データフレームを３つに分割
    df_train = df_observation[:train_num]
    df_validation = df_observation[train_num:train_num+validation_num]
    df_test = df_observation[train_num+validation_num:]
    
    return df_train, df_validation, df_test

def make_validation_test(df,
                         time_step=6, # time_step*10 分間の観測データを使う
                         time_threshold=24, # 単位は hour
                         sample_size_half=50,
                         ):
    df_short = df[df["time to eruption"] <= time_threshold]
    df_long = df[df["time to eruption"] > time_threshold]
    eoo_short = df_short["end of observation"].values
    eoo_long = df_long["end of observation"].values
    eoo_short = list(map(read_data.str_to_datetime, eoo_short))
    eoo_long = list(map(read_data.str_to_datetime, eoo_long))
    time_delta = datetime.timedelta(0, time_threshold*3600)
    eoo_sample = []
#    eoo_sample_short = []*sample_size_half
    for count in range(sample_size_half*2):
        if count % 2==0:
            eoo = random.choice(eoo_short)
        else:
            eoo = random.choice(eoo_long)
        eoo_sample.append(eoo)
        eoo_short = [x_short for x_short in eoo_short if abs(x_short-eoo)<time_delta]
        eoo_long = [x_long for x_long in eoo_long if abs(x_long-eoo)<time_delta]
        
        assert len(eoo_short)+len(eoo_long) > len(df) - time_threshold*6*2 - 1
        count += 1
    return eoo_sample
#    time_to_eruptions = 
    
    
def main():     
    df_train, df_validation, df_test = devide_data(time_step=24*6)
    eoo_sample=make_validation_test(df_validation, slash_dash="dash")
    print(len(eoo_sample))
#    make_model(input_shape=(29,29,144,1))  
 
    
if __name__ == '__main__':
    main()

