import math
import numpy as np
import os
import os.path
import scipy.misc
import tensorflow as tf
import urllib
import gzip
import cPickle as pickle
import PIL.Image
import pandas as pd

from scipy.misc import imsave
from download import *
import image_load_helpers
from celebA_input import inputs
import glob

# def load_celeba_test(path='./CelebA/'):
def CelebA_load(label_data = None, image_paths = None, batch_size = 64, isTrain=True):
    path='./data/CelebA/'

    assert os.path.exists(path + 'is_male.csv')
    assert os.path.isdir(path + 'images/')
    
    if( label_data is None ):
        label_data = np.squeeze((pd.read_csv(path + 'is_male.csv') + 1).astype(np.bool).astype(np.int32).values)
        image_paths = glob.glob(path + 'images/*')
        image_paths.sort()
        return 1 - label_data, image_paths
    
    tot_len = len(label_data)
    test_num = int(tot_len * 0.1)
    if( isTrain ): 
        index = 1 + np.random.choice(tot_len - test_num, batch_size, False)
    else:
        index = 1 + tot_len - test_num + np.random.choice(test_num, batch_size, False)
    
    images = np.array([image_load_helpers.get_image(image_paths[i], 108).reshape([64*64*3]) for i in index])/255.
    labels = label_data[index-1]
    
    return images, labels

def shuffle(images, targets):
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)

def cifar10_load():
    path = './pretrained_models/cifar10/data/cifar10_train/cifar-10-batches-py/'
    batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    data = []
    targets = []
    for batch in batches:
        with open(path + batch, 'rb') as file_handle:
            batch_data = pickle.load(file_handle)
            batch_data['data'] = (batch_data['data'] / 255.0)
            data.append(batch_data['data'])
            targets.append(batch_data['labels'])
    with open(path + 'test_batch') as file_handle:
        batch_data = pickle.load(file_handle)
        batch_data['data'] = (batch_data['data'] / 255.0)
        return np.vstack(data), np.concatenate(targets), batch_data['data'], batch_data['labels']

def MNIST_load():
    filepath = './data/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print ("Couldn't find MNIST dataset in ./data, downloading...")
        urllib.urlretrieve(url, filepath)

    with gzip.open(filepath, 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    tr_image, tr_label = train_data
    ts_image, ts_label = test_data
    shuffle(tr_image, tr_label)
    shuffle(ts_image, ts_label)

    return (tr_image, tr_label, ts_image, ts_label)

def file_exists(path):
    return os.path.isfile(path)

def LeakyReLU(x, alpha=0.1):
    return tf.maximum(x, alpha*x)

def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    nh, nw = rows, int(n_samples/rows) + 1

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))
    
    if X.ndim == 4:
        # BCHW -> BHWC
        if( X.shape[1] == 3 ):
            X = X.transpose(0,2,3,1)
            
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    imsave(save_path, img)

def test_2d():
    it, TRAIN_SIZE, TEST_SIZE = 0, 260520, 2000
    train_data, test_data = [], []
    train_target, test_target = [], []
    while( it < TRAIN_SIZE + TEST_SIZE ):
        x0, y0 = 4 * (np.random.randint(3, size=2) - 1)
        r = np.random.normal(0, 0.5)
        t = np.random.uniform(0, 6.3)
        xy = np.matrix([x0 + (r**2)*math.cos(t), y0 + (r**2)*math.sin(t)])
        #x0, y0 = np.random.uniform(0, 1, size=2)
        #xy = np.matrix([x0 + 1, y0 + 1])
        label = 1

        it = it + 1
        if( it < TRAIN_SIZE ):
            train_data.append(xy)
            train_target.append(label)
        else:
            test_data.append(xy)
            test_target.append(label)

    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)

    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.scatter(np.asarray(train_data[:, 0]).flatten(), np.asarray(train_data[:, 1]).flatten(), s=0.4, c='b', alpha=0.7)

    fig.savefig('train.png')
    plt.close()

    return train_data, train_target, test_data, test_target

# Toy Testset
def swiss_load():
    it, TRAIN_SIZE, TEST_SIZE = 0, 65536, 2000
    train_data, test_data = [], []
    train_target, test_target = [], []
    while( it < TRAIN_SIZE + TEST_SIZE ):
        t = np.random.uniform(0, 10)

        xy = 0.5*np.matrix([t*math.cos(2*t), t*math.sin(2*t)])
        label = int(t < 5)

        it = it + 1
        if( it < TRAIN_SIZE ):
            train_data.append(xy)
            train_target.append(label)
        else:
            test_data.append(xy)
            test_target.append(label)

    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)

    return train_data, train_target, test_data, test_target

def dynamic_load_dataset(batch_size, load_func):
    label_data, image_paths = load_func()
    tot_len = len(label_data)
    
    test_len = int(tot_len*0.1)
    train_len = tot_len - test_len
    
    def train_epoch():
        i = 0
        while(i + batch_size < train_len):
            data, label = load_func(label_data, image_paths, batch_size, isTrain=True)
            yield data, label
            i = i + batch_size
            
    def test_epoch():
        i = 0
        while(i + batch_size < test_len):
            data, label = load_func(label_data, image_paths, batch_size, isTrain=False)
            yield data, label
            i = i + batch_size
            
    return train_epoch, None, test_epoch
    
def load_dataset(batch_size, load_func, dynamic_load = False):
    if( dynamic_load ):
        return dynamic_load_dataset(batch_size, load_func)
        
    train_data, train_target, test_data, test_target = load_func()
    test_size = batch_size

    def train_epoch():
        tot_len = train_data.shape[0]
        i = 0
        #i = np.random.randint(0, batch_size)
        while(i + batch_size < tot_len):
            yield (np.copy(train_data[i:i+batch_size, :]), np.copy(train_target[i:i+batch_size]))
            i = i + batch_size

    def test_epoch():
        tot_len = test_data.shape[0]
        i = 0
        #i = np.random.randint(0, test_size)
        while(i + test_size < tot_len):
            yield (np.copy(test_data[i:i+test_size, :]), np.copy(test_target[i:i+test_size]))
            i = i + batch_size

    return train_epoch, None, test_epoch

def batch_gen(gens, use_one_hot_encoding=False, out_dim=-1, num_iter=-1):
    it = 0
    while (it < num_iter) or (num_iter < 0):
        it = it + 1

        for images, targets in gens():
            if( use_one_hot_encoding ):
                n = len(targets)
                one_hot_code = np.zeros((n, out_dim))
                one_hot_code[range(n), targets] = 1
                yield images, one_hot_code
            else:
                yield images, targets

def save_digit(image_array, image_path):
    reshaped_image = image_array.reshape(28, 28) * 255
    PIL.Image.fromarray(reshaped_image).convert('LA').save(image_path)
