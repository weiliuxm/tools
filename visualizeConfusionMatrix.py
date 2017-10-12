'''
Date : September 2017

Authors : Wei Liu from East China Jiaotong university

Description : code used to compute and plot confusion matrix 

Execution : simply type the following command in a terminal:

   >> python visualizeConfusionMatrix.py gt.csv your_results.csv

NOTE: this code was developed and tested with Python 2.7.6 and Linux (Ubuntu 14.04)

'''

from __future__ import print_function
import csv
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score
import numpy as np
import sys

classes = ['articulated_truck', 'bicycle', 'bus', 'car',
           'motorcycle', 'non-motorized_vehicle', 'pedestrian',
           'pickup_truck', 'single_unit_truck', 'work_van', 'background']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_confusion_matrix(gt_file, res_file, classes):
    '''
     Both 'result' and 'gt' are a python dictionary for which 'image name' is
     the key and 'label' is the value, ex:
           gt['00008854.jpg'] = 'bicycle'
    '''
    gt = {}
    with open(gt_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            gt[row[0]] = row[1]

    results = {}
    with open(res_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            results[row[0]] = row[1]
    y_true = []
    y_pred = []

    for img in gt.keys():
        y_true.append(gt[img])
        if img in results.keys():
            y_pred.append(results[img])
        else:
            print('\nWarning!! you forgot to label ',results[img])
            y_pred.append('error')

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return cm

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage : \n\t python visualizeConfusionMatrix gt.csv your_results.csv")
    else:
        print('Computing a confusion matrix  between ', sys.argv[1], ' and ', sys.argv[2], '\n')
        cnf_matrix = get_confusion_matrix(sys.argv[1], sys.argv[2], classes)

	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes,
        	              title='Confusion matrix, without normalization')

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes, normalize=True,
        	              title='Normalized confusion matrix')

	plt.show()

