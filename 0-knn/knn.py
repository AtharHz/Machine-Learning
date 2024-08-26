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
    neighbors = get_neighbors(train, test_row, num_neighbors)
    outputs = [row[-1] for row in neighbors]
    return max(set(outputs), key=outputs.count)


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


wrong = 0
accuracies = []

train = list(rows)
size = int(len(rows) / k_fold)
indices = np.arange(len(rows))
folds = []

for i in range(k_fold):
    test_indices = indices[i * size: (i + 1) * size]
    train_indices = np.concatenate([indices[:i * size], indices[(i + 1) * size:]])
    folds.append((train_indices, test_indices))
    
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
