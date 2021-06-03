from utility import *
from tensorflow.keras import datasets

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    #print('train_images', train_images) #matrice 3xn
    print('train_labels', train_images) #matrice 1

    #print('test_images', len(test_images))
    print('test_labels', test_labels) #10000
