from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
import glob
import numpy as np
from PIL import ImageFile
from sklearn.manifold import TSNE
import pickle
import csv

ImageFile.LOAD_TRUNCATED_IMAGES = True
model = ResNet50(weights='imagenet')
model.summary()

ResNet50_feature_list = []
num = 0
Images_list = []
for imageFile in glob.glob('samples/20000/Images/*.jpg'):
#for imageFile in glob.glob('data2/add/*.jpg'):
    # current image prog
    #print(imageFile)
    splitted = imageFile.split("/")
    Images_list.append(splitted[len(splitted) - 1])

    img = image.load_img(imageFile, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    ResNet50_feature_list.append(vgg16_feature_np.flatten())
    print(num)
    num = num + 1

# number of clusters we are using
clusters = 2

with open("Features20000.txt", "wb") as fp:
    pickle.dump(ResNet50_feature_list, fp)

ResNet50_feature_list_np = np.array(ResNet50_feature_list)

kmeans = KMeans(n_clusters=clusters, random_state=0).fit(ResNet50_feature_list_np)
labels = kmeans.labels_


# labels = []
tokens = []

for i in range(0, num):
    tokens.append(ResNet50_feature_list_np[i])
#    labels.append(Images_list[i])

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(ResNet50_feature_list_np)

locations = []

for i in range(0, len(new_values)):
    locations.append([])
    locations[i].append("C0")
    locations[i].append([])
    locations[i][1].append(new_values[i][0])
    locations[i][1].append(new_values[i][1])

print(locations)
l = []
print(Images_list)
for i in range(0,num):
    l.append([])
    l[i].append(Images_list[i])
    l[i].append(new_values[i][0])
    l[i].append(new_values[i][1])

with open("Locations20000.txt", "wb") as fp:
    pickle.dump(locations, fp)

start = ['name', 'pca_x', 'pca_y']

with open('L20000.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(start)
    writer.writerows(l)