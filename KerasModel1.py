from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
import glob
import numpy as np
from sklearn import metrics
from PIL import ImageFile

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle


ImageFile.LOAD_TRUNCATED_IMAGES = True
model = ResNet50(weights='imagenet')
model.summary()

ResNet50_feature_list = []
num = 0
Images_list = []
for imageFile in glob.glob('data/train/test2/*.jpg'):

    # current image prog
    # print(imageFile)
    splitted = imageFile.split("/")
    Images_list.append(splitted[len(splitted) - 1])

    img = image.load_img(imageFile, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    ResNet50_feature_list.append(vgg16_feature_np.flatten())
    # print(num)
    num = num + 1

# number of clusters we are using
clusters = 2

# the clusters list, used to calc purity
clusters_list = []
TSNE_list = []
# make the empty list
for i in range(0, clusters):
    clusters_list.append([])
    TSNE_list.append([])

ResNet50_feature_list_np = np.array(ResNet50_feature_list)
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(ResNet50_feature_list_np)
labels = kmeans.labels_
print(labels)

# for each image
for n in range(0, num):
    # add the image to its cluster group
    clusters_list[labels[n]].append(Images_list[n])

Purity = 0
for i in range(0, clusters):
    eyeImage = 0
    heartImage = 0

    for j in range(0, len(clusters_list[i])):
        if "eye" in clusters_list[i][j]:
            eyeImage = eyeImage + 1
            TSNE_list[i].append("eye")
        if "heart" in clusters_list[i][j]:
            heartImage = heartImage + 1
            TSNE_list[i].append("heart")
    if heartImage > eyeImage:
        Purity = Purity + heartImage
    else:
        Purity = Purity + eyeImage


print(TSNE_list)

print(Purity)
print(num)

Purity = float(Purity) / num
print(Purity)

with open("data.txt", "wb") as fp:
    pickle.dump(ResNet50_feature_list_np, fp)
##
# labels = []
# tokens = []
#
# for i in range(0, num):
#     tokens.append(ResNet50_feature_list_np[i])
#     labels.append(Images_list[i])
#
# tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
# new_values = tsne_model.fit_transform(ResNet50_feature_list_np)
#
# x = []
# y = []
# for value in new_values:
#     x.append(value[0])
#     y.append(value[1])
#
# with open("LocationsX.txt", "wb") as fp:
#    pickle.dump(x, fp)
#
#
# with open("LocationsY.txt", "wb") as fp:
#    pickle.dump(y, fp)
##
