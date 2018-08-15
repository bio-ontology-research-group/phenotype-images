from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import KMeans
import glob
import numpy as np
from PIL import ImageFile
from sklearn.manifold import TSNE
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True
model = ResNet50(weights='imagenet')
model.summary()

with open("Features.txt", "rb") as fp:   # Unpickling
    ResNet50_feature_list = pickle.load(fp)

num = 0
old_l = len(ResNet50_feature_list)
Images_list = []
for imageFile in glob.glob('data2/add/*.jpg'):

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
    print(num)
    num = num + 1

# number of clusters we are using
# clusters = 2

ResNet50_feature_list_np = np.array(ResNet50_feature_list)
# kmeans = KMeans(n_clusters=clusters, random_state=0).fit(ResNet50_feature_list_np)
# labels = kmeans.labels_


tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(ResNet50_feature_list_np)

locations = []
loc = len(new_values) - num
for i in range(loc, len(new_values)):
    locations.append([])
    print(i - old_l)
    locations[i - old_l].append("C1")
    locations[i - old_l].append([])
    locations[i - old_l][1].append(new_values[i][0])
    locations[i - old_l][1].append(new_values[i][1])

with open("Locations.txt", "rb") as fp:   # Unpickling
    l2 = pickle.load(fp)
    for i in locations:
        l2.append(i)
with open("Locations.txt", "wb") as fp:
        pickle.dump(l2, fp)

#with open("Locations.txt", "wb") as fp:
 #   pickle.dump(locations, fp)
