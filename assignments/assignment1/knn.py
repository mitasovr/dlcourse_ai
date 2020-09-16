import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]

        # print(num_train[0])
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test][i_train] = np.sum(np.abs(self.train_X[i_train] - X[i_test]))
                pass

        return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            # t = np.zeros(self.train_X.shape, np.int32)
            # p = t + X[i_test]

            # r = np.abs(p - self.train_X)
            r = np.abs(X[i_test] - self.train_X)
            dists[i_test] = np.sum(r, axis=1)


            # dists[i_test] = np.sum(np.abs(X[i_test] - self.train_X), axis=1)
            pass

        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        # num_train = self.train_X.shape[0]
        # num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        # dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!

        # test_temp = np.repeat(X, repeats=num_train, axis=0)
        # train_temp = np.tile(self.train_X, (num_test, 1))
        # # print(a + b)
        # t1 = np.abs(train_temp - test_temp)
        # t2 = np.sum(t1, axis=1)
        # return t2.reshape(num_test, num_train)

        return np.sum(np.abs(X[:, np.newaxis] - self.train_X), axis=-1)

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # argsort - sort array, result will have indexes to original array
            # self.train_y - array of boolean values false - 0, true - 9
            # closest_y - sorted array of true/false, closests - first
            closest_y = self.train_y[np.argsort(dists[i])]
            
            # k_closest_y - k samples
            k_closest_y = closest_y[:self.k]

            # calc count of true and false
            values, counts = np.unique(k_closest_y, return_counts=True)

            # pick most popular option
            pred[i] = values[np.argmax(counts)]
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        for i in range(num_test):
            closest_y = self.train_y[np.argsort(dists[i])]

            k_closest_y = closest_y[:self.k]

            values, counts = np.unique(k_closest_y, return_counts=True)
            pred[i] = values[np.argmax(counts)]
            # pred[i] = values

            # nearest training samples
            pass
        return pred
