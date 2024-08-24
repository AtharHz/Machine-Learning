import math
import csv
import numpy as np
from random import randrange


def euclidean(first, second):
    distance = 0.0
    for i in range(len(first) - 1):
        distance = np.sum((float(first[i]) - float(second[i])) ** 2)
    return math.sqrt(distance)


def get_neighbors(train, test_row, k):
    distances = list()
    for train_row in train:
        dist = euclidean(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def predict(train, test_row, num_neighbors):
    # train = np.array(train[:,:-1])
    # train = list(train)
    neighbors = get_neighbors(train, test_row, num_neighbors)
    outputs = [row[-1] for row in neighbors]
    return max(set(outputs), key=outputs.count)


# def k_fold_validation(rows, length ,k):
#     for i in range(0, k):
#         print(i)
#         test = np.array(rows[i * length:(i + 1) * length])
#         if i == 0:
#             train = rows[((i + 1) * length):]
#         elif i == k - 1:
#             train = rows[:(i * length)]
#         else:
#             train = np.concatenate([rows[:(i * length)], rows[((i + 1) * length):]])
#         return train,test


file = open('./IRIS.csv')
csvreader = csv.reader(file)
header = next(csvreader)
rows = []
for row in csvreader:
    rows.append(row)
file.close()

knn = int(input("Enter the K for K Nearest Neighbor"))
k_fold = int(input("Enter the K for K Fold Validation"))


#this should give Iris-virginica as told in the question
print(predict(rows, [6, 3, 5, 2], knn))

#data set has 3 species, to avoid ties we should choose some number that isn’t a multiple of 3
#k_fold=10 gives the best result Smaller values don't give as good estimates, and larger values don't provide much better results either
# knn = 10
# k_fold = 5

wrong = 0
accuracies = []

# fold_size = len(X) // k
# folds_X = [X[i * fold_size:(i + 1) * fold_size] for i in range(k)]
# folds_y = [y[i * fold_size:(i + 1) * fold_size] for i in range(k)]
train = list(rows)
size = int(len(rows) / k_fold)
indices = np.arange(len(rows))
folds = []
for i in range(k_fold):
    test_indices = indices[i * size: (i + 1) * size]
    train_indices = np.concatenate([indices[:i * size], indices[(i + 1) * size:]])
    folds.append((train_indices, test_indices))
# test = list()
# while len(test) < size:
#     index = randrange(len(train))
#     test.append(train.pop(index))
# X_train, y_train = X[train_indices], y[train_indices]
# X_test, y_test = X[test_indices], y[test_indices]
for train_indices, test_indices in folds:
    train = np.array(rows)[train_indices.astype(int)]
    test = np.array(rows)[test_indices.astype(int)]
    right_answer = 0
    total = 0
    for j in range(len(test)):
        prediction = predict(train, test[j], knn)
        print('Expected %s, Got %s.' % (str(test[j][-1]), prediction))
        if str(test[j][-1]) == prediction:
            right_answer += 1
        else:
            print("Wrong Prediction")
            wrong += 1
        total += 1
    accuracies.append(right_answer / total)
print("\nAccuracies: ", accuracies)
print("Mean Accuracy:", np.mean(accuracies))
