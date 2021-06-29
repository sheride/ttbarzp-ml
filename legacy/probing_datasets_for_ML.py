#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:36:05 2021

@author: elijahsheridan
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(9,))
layer1 = layers.Dense(40, activation='softmax')(inputs)
layer2 = layers.Dense(40, activation='softmax')(layer1)
layer3 = layers.Dense(40, activation='softmax')(layer2)
outputs = layers.Dense(1, activation='softmax')(layer3)
model = keras.Model(inputs=inputs, outputs=outputs, name='sig_bg_bin_class')
model.summary()

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"],
)

predictors = np.load('train_data/pred_m350G.npy')
labels = np.load('train_data/lab_m350G.npy')

history = model.fit(predictors[:100000],
                    labels[:100000],
                    batch_size=1,
                    epochs=100,
                    validation_split=0.2)

#weights = np.array(model.get_weights())
#print(weights)

#print(np.array2string(weights, suppress_small=True, formatter={'float': '{:0.4f}'.format}))

model.evaluate(predictors[50000:51000], labels[50000:51000], verbose=2)

results = np.rint(model.predict(predictors[50000:51000]))
evaluate = (1000 - np.sum(np.abs(results - labels[50000:51000]))) / 1000
print(evaluate)

print(np.sum(results))

