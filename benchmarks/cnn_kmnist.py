#!/usr/bin/env python
# -*- coding: utf-8 -*-

# cnn_kmnist.py
#----------------
# Train a small CNN to identify 10 Japanese characters in classical script
# Based on MNIST CNN from Keras' examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py (MIT License)

from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers
from scipy.ndimage import map_coordinates, gaussian_filter #used for elastic deformation

import argparse
import numpy as np
import os
from utils import load_train_data, load_test_data, load, KmnistCallback
import wandb
from wandb.keras import WandbCallback

# default configuration / hyperparameter values
# you can modify these below or via command line
MODEL_NAME = ""
DATA_HOME = "./dataset" 
BATCH_SIZE = 128
EPOCHS = 100
FILTERS = 32
DROPOUT_1_RATE = 0.25
DROPOUT_2_RATE = 0.5
NUM_CLASSES = 10
#NUM_CLASSES_K49 = 49

# input image dimensions
img_rows, img_cols = 28, 28
# ground truth labels for the 10 classes of Kuzushiji-MNIST Japanese characters 
LABELS_10 =["お", "き", "す", "つ", "な", "は", "ま", "や", "れ", "を"] 
LABELS_49 = ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち",
"つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め"
"も","や","ゆ","よ","ら","り","る","れ","ろ","わ","ゐ","ゑ","を","ん","ゝ"]

def train_cnn(args):
  # initialize wandb logging to your project
  wandb.init(entity="tom", project="kmnist")
  config = {
    "model_type" : "cnn",
    "batch_size" : args.batch_size,
    "num_classes" : args.num_classes,
    "epochs" : args.epochs,
    "filters": args.filters,
    "dropout_1" : args.dropout_1,
    "dropout_2" : args.dropout_2,
  }
  wandb.config.update(config)

  # Load the data form the relative path provided
  x_train, y_train = load_train_data(args.data_home)
  x_test, y_test = load_test_data(args.data_home)

  # reshape to channels last
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  if args.quick_run:
    MINI_TR = 6000
    MINI_TS = 1000
    x_train = x_train[:MINI_TR]
    y_train = y_train[:MINI_TR]
    x_test = x_test[:MINI_TS]
    y_test = y_test[:MINI_TS]
 
  N_TRAIN = len(x_train)
  N_TEST = len(x_test)
  wandb.config.update({"n_train" : N_TRAIN, "n_test" : N_TEST})
  print('{} train samples, {} test samples'.format(N_TRAIN, N_TEST))

  # Convert class vectors to binary class matrices
  y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
  
  def conv2d(filters, kernel_size=(3, 3), **kwargs):
    return layers.Conv2D(filters, kernel_size=kernel_size, kernel_initializer='lecun_normal', padding='same', use_bias=False, **kwargs)
  
  def dense(filters, **kwargs):
    return layers.Dense(filters, kernel_initializer='lecun_normal', use_bias=False, **kwargs)
  
  def act(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
  
  def block(x_init, filters, down=False):
    strides = 2 if down else 1
    
    a = conv2d(filters, kernel_size=1, strides=strides)(x_init)
    a = act(a)
    
    b = layers.AveragePooling2D(pool_size=3, strides=strides, padding='same')(x_init)
    b = conv2d(filters, kernel_size=1)(b)
    b = act(b)
    
    c = conv2d(filters, kernel_size=(1, 3))(x_init)
    c = conv2d(filters, kernel_size=(3, 1), strides=strides)(c)
    c = act(c)
    
    d = conv2d(filters, kernel_size=(1, 5))(x_init)
    d = conv2d(filters, kernel_size=(5, 1), strides=strides)(d)
    d = act(d)
    
    return layers.Concatenate()([a, b, c, d])
  
  def stack(x_init, filters, count):
    x = x_init
    
    blocks = []
    for i in range(count):
      x = block(x, filters, down=(i==0))
      blocks.push(x)
    
    return blocks[-1]
#     return Add()(blocks)

  # Build model
  input = layers.Input(input_shape)
  
  x = layers.GaussianNoise(0.1)(input)
  
  x = conv2d(args.filters, kernel_size=(7, 7))(x)
  x = act(x)
  
  x = stack(x, args.filters*2, 3)
  
  x = layers.SpatialDropout2D(args.dropout_1)(x)
  x = layers.GaussianNoise(0.1)(x)
  
  x = stack(x, args.filters*4, 2)
  
  x = layers.SpatialDropout2D(args.dropout_1)(x)
  x = layers.GaussianNoise(0.1)(x)
  
  x = stack(x, args.filters*8, 2)
  
  x = layers.SpatialDropout2D(args.dropout_2)(x)
  x = layers.GaussianNoise(0.1)(x)
  
  x = layers.GlobalAveragePooling2D()(x)
  x = dense(args.num_classes)(x)
  output = layers.Activation('softmax')(x)
  
  model = tf.keras.Model(input, output)
  model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
  
  def lr_schedule(epoch):
    lr = 1e-3
    final_lr = 1e-5
    base = np.power(final_lr / lr, 1. / args.epochs)
    return lr * base**epoch
  
  # adapted from https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
  def deform_grid(X, sigma_range=3, points=3):
    sigma = np.random.uniform(0, sigma_range)
    
    orig_shape = X.shape
    shape = (orig_shape[0], orig_shape[1])
    X = X.reshape(shape)

    coordinates = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    xi=np.meshgrid(np.linspace(0,points-1,shape[0]), np.linspace(0,points-1,shape[1]), indexing='ij')
    grid = [points, points]

    for i in range(2):
      yi=np.random.randn(*grid)*sigma
      y = map_coordinates(yi, xi, order=3).reshape(shape)
      coordinates[i]=np.add(coordinates[i],y) #adding the displacement

    return map_coordinates(X, coordinates, order=3).reshape(orig_shape)
  
  def deform_pixel(X, alpha_range=15, sigma=5):
    alpha = np.random.uniform(0, alpha_range)
    
    orig_shape = X.shape
    shape = (orig_shape[0], orig_shape[1])
    X = X.reshape(shape)
    
    dx = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = x+dx, y+dy

    return map_coordinates(X, indices, order=3).reshape(orig_shape)
  
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    shear_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    preprocessing_function=deform_pixel,
  )

  model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=args.batch_size),
            epochs=args.epochs,
            steps_per_epoch=len(x_train)//args.batch_size,
            verbose=1,
            use_multiprocessing=True,
            validation_data=(x_test, y_test),
            callbacks=[
              KmnistCallback(),
              WandbCallback(data_type="image", labels=LABELS_10),
              tf.keras.callbacks.LearningRateScheduler(lr_schedule)
            ])

  train_score = model.evaluate(x_train, y_train, verbose=0)
  test_score = model.evaluate(x_test, y_test, verbose=0)
  print('Train loss:', train_score[0])
  print('Train accuracy:', train_score[1])
  print('Test loss:', test_score[0])
  print('Test accuracy:', test_score[1])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default=MODEL_NAME,
    help="Name of this model/run (model will be saved to this file)")
  parser.add_argument(
    "--data_home",
    type=str,
    default=DATA_HOME,
    help="Relative path to training/test data")
  parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="batch size")
  parser.add_argument(
    "--dropout_1",
    type=float,
    default=DROPOUT_1_RATE,
    help="dropout rate for first dropout layer")
  parser.add_argument(
    "--dropout_2",
    type=float,
    default=DROPOUT_2_RATE,
    help="dropout rate for second dropout layer")
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=EPOCHS,
    help="number of training epochs (passes through full training data)")
  parser.add_argument(
    "--filters",
    type=int,
    default=FILTERS,
    help="base number of filters")
  parser.add_argument(
    "--num_classes",
    type=int,
    default=NUM_CLASSES,
    help="number of classes (default: 10)")
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
  parser.add_argument(
    "--quick_run",
    action="store_true",
    help="train quickly on a tenth of the data")   
  args = parser.parse_args()

  # easier testing--don't log to wandb if dry run is set
  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'

  # create run name from command line
  if args.model_name:
    os.environ['WANDB_DESCRIPTION'] = args.model_name

  train_cnn(args)

