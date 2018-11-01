#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:24:58 2018

@author: muratamasaki
"""

from keras.layers import Input, Dense, Conv3D, Flatten
from keras.models import Model


def make_model(input_shape,
               ):
    input_img = Input(shape=input_shape)
    x = Conv3D(filters=2, kernel_size=(2,2,2), padding="valid", activation="relu")(input_img)
    output_conv = Conv3D(filters=2, kernel_size=(2,2,2), padding="valid", activation="relu")(x)
    
    x = Flatten()(output_conv)
    x = Dense(256, activation="relu")(x)
    output = Dense(1, activation="relu")(x)
    
    model = Model(input_img, output)
    
    model.summary()
    
def main():     
    make_model(input_shape=(29,29,144,1))  
    
if __name__ == '__main__':
    main()

