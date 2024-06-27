from typing import Any
from numpy import *
import csv
from PIL import Image
# import util

labels = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
'bottle', 'bowl', 'can', 'cup', 'plate',
'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
'clock', 'keyboard', 'lamp', 'telephone', 'television',
'bed', 'chair', 'couch', 'table', 'wardrobe',
'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
'bear', 'leopard', 'lion', 'tiger', 'wolf',
'bridge', 'castle', 'house', 'road', 'skyscraper',
'cloud', 'forest', 'mountain', 'plain', 'sea',
'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
'crab', 'lobster', 'snail', 'spider', 'worm',
'baby', 'boy', 'girl', 'man', 'woman',
'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']

class C100Dataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = None  # X (data) of training set.
    tr_y = None  # Y (label) of training set.
    ts_x = None # X (data) of test set.
    ts_y = None # Y (label) of test set.

    def __init__(self, train_filename, test_filename):
        ## read the csv for dataset (cifar100.csv, cifar100_lt.csv or cifar100_nl.csv), 
        # 
        # Format:
        #   image file path,classname
        
        ### TODO: Read the csv file and make the training and testing set

        with open(train_filename, 'r') as f:
            reader = csv.reader(f)
            result = [row for i, row in enumerate(reader) if i < 49999] 
            # print(result)

        arr = array(result)
        train_file, y_tr = arr.T
        # print(train_file[0:5], y_train[0:5])

        with open(test_filename, 'r') as f:
            reader = csv.reader(f)
            result = [row for i, row in enumerate(reader)] 

        arr = array(result)
        # print(arr)
        test_file, y_ts = arr.T

        y_train = zeros(len(y_tr))
        y_test = zeros(len(y_ts))

        for i, y in enumerate(y_tr):
          y_train[i] = int(labels.index(y, 0, 100))

        # print(y_train)

        y_train = y_train.astype(int8)
        
        for i, y in enumerate(y_ts):
          y_test[i] = int(labels.index(y, 0, 100))

        y_test = y_test.astype(int8)
        # print(x, y)
        ## YOUR CODE HERE

        ### TODO: assign each dataset. 

        x_train = array([array(Image.open('/content/Yonsei-vnl-coding-assignment-vision-48hrs/dataset/'+fname)) for fname in train_file])
        x_test = array([array(Image.open('/content/Yonsei-vnl-coding-assignment-vision-48hrs/dataset/'+fname)) for fname in test_file])

        print(x_train.shape, x_test.shape)

        # x = x.reshape((-1, 1, 3072))
        # y = y.reshape((-1, 1))
        # # print(x)
        # print(x.shape, y.shape)

        self.tr_y = zeros((y_train.size, 100), dtype=int)
        self.ts_y = zeros((y_test.size, 100), dtype=int)

        self.tr_x = x_train  ### TODO: YOUR CODE HERE
        self.tr_y[arange(y_train.size),y_train] = 1  ### TODO: YOUR CODE HERE
        self.ts_x = x_test ### TODO: YOUR CODE HERE
        self.ts_y[arange(y_test.size),y_test] = 1 ### TODO: YOUR CODE HERE

        self.tr_x = self.tr_x.transpose((0, 3, 1, 2))
        self.ts_x = self.ts_x.transpose((0, 3, 1, 2))

        print(self.tr_y[0][y_train[0]], y_train[0])

        self.tr_data = [self.tr_x, self.tr_y]
        self.ts_data = [self.ts_x, self.ts_y]
        # self.tr_data = concatenate((x[:tr], y[:tr]), axis=0)
        # self.ts_data = concatenate((x[tr:], y[tr:]), axis=0)

        # print(self.tr_data.shape)
        
    def getDataset(self):
        # return [self.tr_x, self.tr_y, self.ts_x, self.ts_y]
        return self.tr_data, self.ts_data

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_samples = len(dataset[0])
        self.indexes = list(range(self.num_samples))
        self.current_batch = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_batch * self.batch_size >= self.num_samples:
            self.current_batch = 0
            raise StopIteration
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min((self.current_batch + 1) * self.batch_size, self.num_samples)

        batch_indexes = self.indexes[start_idx:end_idx]
        inputs = self.dataset[0][batch_indexes]
        labels = self.dataset[1][batch_indexes]

        batch_data = (inputs, labels, batch_indexes)

        self.current_batch += 1

        return batch_data
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
# def main():   
#     # data = C100Dataset('data/cifar100_nl.csv', 0.8)

#     train, val = C100Dataset('data/cifar100_nl.csv', 0.8).getDataset()

#     trainloader = DataLoader(train, 32)
#     valloader = DataLoader(val, 32)

#     # print(len(trainloader), len(valloader))

#     # for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
#     #     if batch_idx == 0:
#     #         print(inputs, targets, indexes)

# if __name__ == '__main__':
#     main()
