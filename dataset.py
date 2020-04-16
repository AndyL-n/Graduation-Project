import scipy.sparse as sp
import numpy as np

class Dataset(object):
    def __init__(self, path):
        self.trainMatrix = self.load_rating_file_as_matrix(path + "train.rating")
        self.testRatings = self.load_test_file_as_list(path + "test.rating")
        self.testNegatives = self.load_negative_file(path + "test.negative")
        print(len(self.testRatings),len(self.testNegatives))
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_test_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            line = line.rstrip("\n")
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                ratingList.append([user, item, rating])
                line = f.readline()
                line = line.rstrip("\n")
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            line = line.rstrip("\n")
            while line != None and line != "":
                arr = line.split("\t")
                print(arr)
                negatives = []
                n = len(arr)
                for x in arr[1:n - 1]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
                line = line.rstrip("\n")
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        # Read .rating file and Return dok matrix.
        # The first line of .rating file is: num_users\t num_items
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            line = line.rstrip("\n")
            # print(line)
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
                line = line.rstrip("\n")

        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            line = line.rstrip("\n")
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                # if (rating > 0):
                mat[user, item] = rating
                # print('user%d\t  item %d\t  rating%f' % (user,item,rating))
                line = f.readline()
                line = line.rstrip("\n")
        return mat
