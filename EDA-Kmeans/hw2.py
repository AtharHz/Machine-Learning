import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# to ignore warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


data = pd.read_csv("Live.csv")

print("Head(First 5 Rows)")
print(data.head())

print("\n\nTail(Last 5 Rows)")
print(data.tail())

print("\n\nNon Unique")
print(data.nunique())

print("\n\nInformation")
print(data.info())

print("\n\nDuplicated Sum")
print(data.duplicated().sum())

print("\n\nIs Null Sum")
print(data.isnull().sum())

print("\n\nDescribe")
print(data.describe().T)

print("Data Types")
print(data.dtypes)

data = data.dropna(axis=1, how='all')
# data = data.dropna(inplace=True)

print("\n\nAfter Deleting Null Columns")
print(data.info())

cat_cols = data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()

print("\n\nCategorical Variables:")
print(cat_cols)

print("\n\nNumerical Variables:")
print(num_cols)

nr_data = data.copy()
print("Checking Categorical Columns")
for x in cat_cols:
    column_uniq = len(pd.unique(data[x]))/len(data[x])
    print("Uniqueness: ", column_uniq)
    print("Column Name: ", x)
    if column_uniq > 0.5:
        nr_data = nr_data.drop([x], axis=1)
        # data = data.drop(['Column4'], axis=1)
        # print(column_uniq)
    else:
        nr_data[x] = nr_data[x].astype('category')
        nr_data[x] = nr_data[x].cat.codes

# scaler = StandardScaler()
# print(nr_data.tail())
# scaler = Normalizer()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(nr_data)
# scaled_df = pd.DataFrame(scaled_data, columns=nr_data.columns)


wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
identified_clusters = kmeans.fit_predict(scaled_data)
print("\n\nIdentified Clusters", identified_clusters)
