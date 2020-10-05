import numpy as np
import tensorflow as tf
import random
import os
import config as cfg

def tf_normalize(data):
    return tf.math.divide(data, tf.math.reduce_max(tf.math.abs(data)))

def FCN_output(xc,yc,zc):
    x = np.reshape(np.linspace(5500,8500,88), (88,1,1))
    y = np.reshape(np.linspace(3500,6100,48), (1,48,1))
    z = np.reshape(np.linspace(1400,3600,40), (1,1,40))
    xc = np.round(xc)
    yc = np.round(yc)
    zc = np.round(zc)
    fcn_out = np.exp(-((x-xc)**2/(2*200**2)+(y-yc)**2/(2*200**2)+(z-zc)**2/(2*200**2)))
    fcn_out = fcn_out/np.max(fcn_out)

    return fcn_out

def tf_FCN_output(xc,yc,zc):
  [fcn,] = tf.py_function(FCN_output, [xc,yc,zc], [tf.float32])
  fcn.set_shape((88,48,40))

  return fcn

def time_shift(data, z):
    negative_time_shifts = np.array([-100, -150, -200, -240, -270, -310])
    positive_time_shifts = np.array([550, 500, 450, 400, 350, 300])
    ind_label = np.argmax(z)
    t_shift = np.random.randint(negative_time_shifts[ind_label], positive_time_shifts[ind_label])
    data = np.roll(data, t_shift, axis=0)

    if t_shift < 0:
        data[t_shift:, :] = 0
    elif t_shift > 0:
        data[:t_shift, :] = 0

    return data

def tf_time_shift(data, z):
    data_shape = data.shape
    [data,] = tf.py_function(time_shift, [data, z], [tf.float32])
    data.set_shape(data_shape)
    return data

def station_dropout(data):
    n_drop = random.randrange(1,10,1)
    drop_stations = np.random.choice(np.arange(96), n_drop, replace=False)
    mask = np.ones((1401, 96))
    mask[:, drop_stations] *= 0
    data = np.multiply(data, mask)
    return data

def tf_station_dropout(data):
    data_shape = data.shape
    [data,] = tf.py_function(station_dropout, [data], [tf.float32])
    data.set_shape(data_shape)
    return data

def add_field_noise(data):
    n_noise = random.randrange(10,41)
    noise_stations = np.random.choice(cfg.field_noise.shape[1], n_noise, replace=False)
    data_stations = np.random.choice(data.shape[1], n_noise, replace=False)
    mask = np.zeros((data.shape[0], data.shape[1]))
    mask[:,data_stations] = np.random.uniform(0.3, 1.0, size=n_noise)*cfg.field_noise[:,noise_stations]
    data = tf_normalize(data)
    data = np.add(data,mask)
    data = data/np.max(np.abs(data))

    return data

def tf_field_noise(data):
    data_shape = data.shape
    [data,] = tf.py_function(add_field_noise, [data], [tf.float32])
    data.set_shape(data_shape)
    return data

def gaussian_noise(data):
    noise_stations = np.random.choice(cfg.gauss_noise.shape[1], 96, replace=False)
    random_noise = cfg.gauss_noise[:,noise_stations]
    t_shift = np.random.randint(0,96)
    random_noise = np.roll(random_noise, t_shift, axis=1)
    t_shift = np.random.randint(0,1401)
    random_noise = np.roll(random_noise, t_shift, axis=0)
    random_noise = np.random.uniform(0.1,0.5, 96)*random_noise[:,:96]
    data = data/np.max(np.abs(data))
    data = data+ random_noise
    data = data/np.max(np.abs(data))
    return data

def tf_gaussian_noise(data):
    data_shape = data.shape
    [data,] = tf.py_function(gaussian_noise, [data], [tf.float32])
    data.set_shape(data_shape)
    return data
