#### IMPORT ####
import numpy as np                                            # Numpy
import keras
from keras.models import Sequential         # This is simply a linear stack of neural network layers, and it's perfect for feed-forward CNN
from keras.layers import Dense, Dropout, Activation, Flatten  # Core layers: These are the layers that are used in almost any NN
from keras.layers import Convolution2D, MaxPooling2D          # CNN layers: These will help us efficiently train on image data
from keras.utils import np_utils, plot_model                  # Utilities: This will help us transform our data later
from keras.models import load_model                           # To save the model on json
from matplotlib import pyplot as plt                          # Plotting
from PIL import Image                                         # Image Import
import os                                                     # Files and Folder Management
from sklearn.utils import shuffle                             # To shuffle data before to enter it in the NN
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, precision_recall_fscore_support, precision_recall_curve, average_precision_score, accuracy_score
import itertools
import pydot_ng as pydot
import graphviz

def Transform_Images_Matrix(folder, Tag):
    path = 'C:/Users/user/Desktop/Data Incubator/Capstone Project/Py Files/data/test/' + folder
    files = os.listdir(path)
    X = []
    list_file_names = []
    print('------- ' + folder + ' ---------')
    for fname in files:
        try:
            temp = np.array(Image.open(path + '/' + fname))
            #print(fname, temp.shape)
            if(temp.shape[2] == 3):
                X.append(np.array(Image.open(path + '/' + fname).resize((150,150))))
                list_file_names.append(fname)
        except:
            continue             
    Pixels = np.array(X)
    #print(Pixels.shape)
    if(Tag):
        Labels = np.repeat(1, Pixels.shape[0])
    else:
        Labels = np.repeat(0, Pixels.shape[0])
    print(Pixels.shape, Labels.shape)
    return Pixels, Labels

def Preprocess(X, Y):
    X = X.astype('float32')                           # Transform to float32
    X /= 255                                          # Normalize Data in range [0,1]
    Y = np_utils.to_categorical(Y, 2)                 # One-hot Encoding: 1 column per class
    return X, Y

def Plot_Confusion_Matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    
    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.
    
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
    
def Plot_Recall_Precision(y_test, y_score):
    precision, recall,_ = precision_recall_curve(y_test, y_score)
    
    average_precision = average_precision_score(y_test, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

    plt.step(recall, precision, color='r', alpha=0.1,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.1,
                     color='r')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

def main():
    np.random.seed(123456789)                                           # For reproducibility
    class_names = ['Non Nude', 'Nude']
    print('......... Concatenating .........')
    Yes_X_test, Yes_Y_test =  Transform_Images_Matrix('nude', True)
    No_X_test, No_Y_test = Transform_Images_Matrix('non_nude', False)
    
    X_test = np.concatenate((Yes_X_test, No_X_test))
    Y_test = np.concatenate((Yes_Y_test, No_Y_test))
    
    Y_test_copy = Y_test
    print('......... Preprocessing .........')
    X_test, Y_test = Preprocess(X_test, Y_test)

    print('......... Loading Model .........')
    model = load_model("model.h5")
    #plot_model(model, to_file='model.png', show_shapes = False, show_layer_names = True, rankdir='TB')
    print("Loaded model from disk")
    
    print('......... Testing .........')
    print(X_test.shape, Y_test.shape)
    score = model.evaluate(X_test, Y_test, verbose=1)            # Evaluate model on test data
    
    predictions = model.predict(X_test)
    print("------------ PRECISION RECALL CURVE -----------------")
    Y_pred = predictions.argmax(axis = -1)
    #Y_pred = []
    #for p in predictions:
        #if p[1] < 0.6:
            #Y_pred.append(0)
        #else:
            #Y_pred.append(1)
    print("Accuracy:", accuracy_score(Y_pred, Y_test_copy))
    print("Precision:",precision_recall_fscore_support(Y_pred, Y_test_copy)[0])
    print("Recall:",precision_recall_fscore_support(Y_pred, Y_test_copy)[1])
    print("F_Score:",precision_recall_fscore_support(Y_pred, Y_test_copy)[2])
    
    precision, recall, thresholds = precision_recall_curve(Y_pred, Y_test_copy)
    for p, r, t in zip(precision, recall, thresholds):
        print(p, r, t)

    
    #for i in range(0, len(Y_test_copy)):
        #if Y_pred[i] != Y_test_copy[i]:
            #print(i)
    
    #print(predictions*100)
    cnf_matrix = confusion_matrix(Y_test_copy, Y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    Plot_Confusion_Matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    Plot_Confusion_Matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    
    #Plot Precision Recall
    plt.figure()
    print(Y_test[:,1], predictions[:,1])
    Plot_Recall_Precision(Y_test[:,1], predictions[:,1])
    plt.show()

main()