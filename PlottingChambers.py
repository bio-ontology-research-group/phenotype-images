import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle


with open("Locations.txt", "rb") as fp:  # Unpickling
    locations = pickle.load(fp)

plt.figure(figsize=(20, 16))
for i in range(0, len(locations)):
    plt.scatter(locations[i][1][0], locations[i][1][1], color=locations[i][0])
    # plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.savefig("plot4")