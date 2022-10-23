import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn import svm

import warnings
warnings.filterwarnings("ignore")
plt.style.use('bmh')

# a simple timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        run_time = end-start
        print("Finished!")
        return run_time, result
    return wrapper

# svm to quickly test out the results
def do_svm(data, label):
    val_index = int(data.shape[0]/5)*4
    clf = svm.SVC(kernel='rbf')
    clf.fit(data[:val_index], label[:val_index])
    predictions = clf.predict(data[val_index:])
    accuracy = np.where(predictions==label[val_index:], 1, 0)
    accuracy = np.sum(accuracy)/accuracy.shape[0]*100
    return accuracy

# data and labels
digits = load_digits()
data = digits['data']/255
labels = digits['target']

# data visualization
plt.figure(0, figsize=(9,9))
plt.clf()
plt.suptitle("Data")
for row in range(3):
    for col in range(3):
        plt.subplot(3,3,row*3+col+1)
        plt.imshow(data[np.random.randint(0,labels.shape[0])].reshape(8,8))
        plt.axis('off')
plt.tight_layout()
plt.show()

# colors for each label
class_colors = np.zeros((labels.size, 3))
unique_labels = np.unique(labels)
how_many_classes = unique_labels.size
for i in range(how_many_classes):
    class_colors[labels==unique_labels[i]] = sns.color_palette()[i]

# figure
plt.figure(1, figsize=(18,6))
plt.clf()
plt.suptitle(f"Accuracy Before: {round(do_svm(data, labels), 1)}%")

# PCA
@timer
def do_pca(data):
    pca = PCA(n_components=2)
    result = pca.fit_transform(data)
    return result
run_time, result = do_pca(data)
accuracy = do_svm(result, labels)

plt.subplot(131)
plt.title(f'PCA ({round(run_time, 3)}s, {round(accuracy, 1)}%)')
plt.scatter(result[:,0], result[:,1], color=class_colors)

# t-SNE
@timer
def do_tsne(data):
    tsne = TSNE(n_components=2)
    result = tsne.fit_transform(data)
    return result
run_time, result = do_tsne(data)
accuracy = do_svm(result, labels)

plt.subplot(132)
plt.title(f't-SNE ({round(run_time, 3)}s, {round(accuracy, 1)}%)')
plt.scatter(result[:,0], result[:,1], color=class_colors)

# UMAP
@timer
def do_umap(data):
    umap = UMAP(n_components=2)
    result = umap.fit_transform(data)
    return result
run_time, result = do_umap(data)
accuracy = do_svm(result, labels)

plt.subplot(133)
plt.title(f'UMAP ({round(run_time, 3)}s, {round(accuracy, 1)}%)')
plt.scatter(result[:,0], result[:,1], color=class_colors)

# show
plt.tight_layout()
plt.show()