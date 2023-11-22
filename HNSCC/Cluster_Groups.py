from natsort import natsorted
import os
import shutil 
import nibabel as nib 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

data = pd.read_csv('C:/Users/poppy/Documents/HN_Atlas/HNSCC/HNSCC_Characteristics.csv', header = 0)
#print(data['Volume'])
#print(data.head)
data['Volume'] = (data['Volume']- data['Volume'].min())/(data['Volume'].max()- data['Volume'].min())
data['Height'] = (data['Height']- data['Height'].min())/(data['Height'].max()- data['Height'].min())
data['minima'] = (data['minima']- data['minima'].min())/(data['minima'].max()- data['minima'].min())
data['a'] = (data['a']- data['a'].min())/(data['a'].max()- data['a'].min())
data['b'] = (data['b']- data['b'].min())/(data['b'].max()- data['b'].min())
data['c'] = (data['c']- data['c'].min())/(data['c'].max()- data['c'].min())


df = data.iloc[:, 1:7].to_numpy()
#print(data.iloc[:, 1:7].head)

pca = PCA()
pca.fit(df)
print(pca.explained_variance_ratio_)

#plt.plot(range(1,7), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
#plt.xlabel('Number of Features')    
#plt.ylabel('Explained Variance')
#plt.show()

'''

inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', random_state=42)
    kmeans.fit(df[:, 0:3])
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

'''

opt_no_clusters = 4

kmeans6 = KMeans(n_clusters=opt_no_clusters, init = 'k-means++', random_state=42)
kmeans6_predictions = kmeans6.fit_predict(df[:, 0:3])
kmeans6_fitted = kmeans6.fit(df)
kmeans6_sqrddistance = kmeans6_fitted.transform(df)**2
#print(kmeans6_sqrddistance)
data['Predictions'] = kmeans6_predictions
Scores = kmeans6_fitted.inertia_
data['Score'] = Scores

#for example in np.arange(0,172):  
#    data['SqrtDistance'][example] = np.round(np.min(kmeans6_sqrddistance[example]))


#-----------------
# sort out folders 
path = 'D:/HNSCC/ARCHIVE/CLUSTERING/'
shutil.rmtree(path)
if not os.path.exists(path):
    os.mkdir(path)

for i in range(0, opt_no_clusters):
    folder = path + 'GROUP_' + str(i+1)
    if not os.path.exists(folder):
        os.mkdir(folder)


for group in np.arange(0,opt_no_clusters):

    Group = data.loc[(data['Predictions'] == group)]
    print(Group)
    #Group = Group.sort_values(by='SqrtDistance', ascending=True)

    for patient in Group['Patient']:

        Score = data.loc[(data['Predictions'] == group) & (data['Patient'] == patient)]['Score']

        org_slice_location = 'D:/HNSCC/ARCHIVE/2023_10_08_ALIGNED/SLC_' + str(patient) + '.png'
        new_slice_location = path + 'GROUP_' + str(group+1) + '/SLC_' + str(patient)+'.png'
        if os.path.exists(org_slice_location):
            shutil.copy(org_slice_location, new_slice_location)


colors = ['red', 'black', 'green', 'orange']
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for group in np.arange(0,opt_no_clusters):

    data_ = data.loc[(data['Predictions'] == group)]
    ax.scatter(data_.iloc[:,1], data_.iloc[:,2], data_.iloc[:,3], color = colors[group])


ax.set_xlabel('Volume')
ax.set_ylabel('Height')
ax.set_zlabel('a')
plt.show()