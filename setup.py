#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:36:17 2021

@author: elijahsheridan
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

path = '/Users/elijahsheridan/MG5_aMC_v2_6_5/b_meson_pheno/ttbarzp-ml/'
names = ['sig_m350GeV', 'sig_m1TeV', 'bg']

def import_data():
    sig350G, sig1TeV, bg = [np.loadtxt(path + name + '.dat') for name in names]

    for name, data in zip(names, [sig350G, sig1TeV, bg]):
        np.save(name, data)

#sig350G, sig1TeV, bg = [np.load(name + '.npy') for name in names]

"""

"""

#names = ['sig_m350GeV', 'sig_m1TeV', 'bg']
#data = [np.load(name + '.npy') for name in names]
#
#sig1, sig2, bg = [
#        (elem - np.mean(elem, axis=0)) / np.std(elem, axis=0) for elem in data]

inputs = keras.Input(shape=(9,))
layer1 = layers.Dense(40, activation='softmax')(inputs)
layer2 = layers.Dense(20, activation='softmax')(layer1)
outputs = layers.Dense(1, activation='softmax')(layer2)
model = keras.Model(inputs=inputs, outputs=outputs, name='sig_bg_bin_class')
model.summary()

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

train = np.load('train1.npy')

train_pred = train[:,:-1]
train_label = train[:,-1]

history = model.fit(train_pred[:500000],
                    train_label[:500000].reshape((500000,1)), batch_size=50,
                    epochs=20, validation_split=0.2)