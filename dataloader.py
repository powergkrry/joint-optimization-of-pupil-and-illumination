import numpy as np


def shuffle_data(data_to_shuffle, random_seed):
    s = np.arange(data_to_shuffle.shape[0])
    np.random.seed(random_seed)
    np.random.shuffle(s)
    data_to_shuffle = data_to_shuffle[s,]
    
    return data_to_shuffle

def load_data(data, random_seed):
    # without illumination
    if data == "center":
        train_x = np.load('./malaria_center_train_x.npy')
        test_x = np.load('./malaria_center_test_x.npy')
    # with illumination
    elif data == "illumination":
        train_x = np.load('./malaria_illumination_train_x.npy')
        test_x = np.load('./malaria_illumination_test_x.npy')
    else:
        sys.exit(1)
    
    train_y = np.load('./malaria_train_y.npy')
    test_y = np.load('./malaria_test_y.npy')
    
    return shuffle_data(train_x, random_seed), shuffle_data(train_y, random_seed), shuffle_data(test_x, random_seed), shuffle_data(test_y, random_seed)