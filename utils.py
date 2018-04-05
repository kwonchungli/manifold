
import numpy as np
import math
import scipy.misc
from scipy.misc import imsave

def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
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
    it, TRAIN_SIZE, TEST_SIZE = 0, 65536, 2000
    train_data, test_data = [], []
    train_target, test_target = [], []
    while( it < TRAIN_SIZE + TEST_SIZE ):
        x0, y0 = 1 * (np.random.randint(10, size=2) - 5)
        r = np.random.normal(0, 0.1)
        t = np.random.uniform(0, 6.3)
        xy = np.matrix([x0 + (r**2)*math.cos(t), y0 + (r**2)*math.sin(t)])
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
    it, TRAIN_SIZE, TEST_SIZE = 0, 32768, 2000
    train_data, test_data = [], []
    train_target, test_target = [], []
    while( it < TRAIN_SIZE + TEST_SIZE ):
        t = np.random.uniform(0, 10)
        
        xy = 0.1*np.matrix([t*math.cos(t), t*math.sin(t)])
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
    
    return train_data, train_target, test_data, test_target
    
def load_dataset(batch_size, load_func):
    train_data, train_target, test_data, test_target = load_func()
    test_size = batch_size
    
    def train_epoch():
        tot_len = train_data.shape[0]
        i = np.random.randint(0, batch_size)
        while(i + batch_size < tot_len):
            yield (np.copy(train_data[i:i+batch_size, :]), np.copy(train_target[i:i+batch_size]))
            i = i + batch_size
    
    def test_epoch():
        tot_len = test_data.shape[0]
        i = np.random.randint(0, test_size)
        while(i + test_size < tot_len):
            yield (np.copy(test_data[i:i+test_size, :]), np.copy(test_target[i:i+test_size]))
            i = i + batch_size

    return train_epoch, None, test_epoch

def batch_gen(gens, use_one_hot_encoding=False, out_dim=-1):
    while True:
        for images, targets in gens():
            if( use_one_hot_encoding ):
                n = len(targets)
                one_hot_code = numpy.zeros((n, out_dim))
                one_hot_code[range(n), targets] = 1
                yield images, one_hot_code
            else:    
                yield images, targets