import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


dataset = pd.read_csv('Breast_Cancer.csv')
dataset.head()
print("Cancer data set dimensions : {}".format(dataset.shape))

cat_cols = dataset.select_dtypes(include=['object']).columns
nr_data = dataset.copy()
for x in cat_cols:
    nr_data[x] = nr_data[x].astype('category')
    nr_data[x] = nr_data[x].cat.codes
# l_encoder = LabelEncoder()
# X = l_encoder.fit_transform(X)
X = nr_data.iloc[:, 0:15].values
Y = nr_data.iloc[:, 15].values
# print(X)
# print("//////////////")
# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# print(X_train)
# print("////////////////")
# print(X_test)
# print("////////////////")
# print(Y_train)
# print("////////////////")
# print(Y_test)
# print("////////////////")

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# classifier = GaussianNB()
# classifier.fit(X_train, Y_train)

classifier = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=7, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None
                                    , random_state=42, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                     class_weight=None, ccp_alpha=0.02)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
# cm = confusion_matrix(Y_test, Y_pred)
print(classification_report(Y_test, Y_pred))
