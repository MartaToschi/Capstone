from keras.preprocessing import image
from PIL import ImageFile
from matplotlib import pyplot as plt
import numpy as np
from keras.models import model_from_json                      # To save the model on json
from keras.applications.imagenet_utils import decode_predictions
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, precision_recall_fscore_support, precision_recall_curve, average_precision_score, accuracy_score
import itertools
import pydot_ng as pydot
import graphviz
from keras.models import load_model                           # To save the model on json


ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    np.random.seed(123456789)
    class_names = ['Non Nude', 'Nude']
    
    test_datagen = image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_generator = test_datagen.flow_from_directory(
            'data/test',
            target_size=(150, 150),
            shuffle=False,
            class_mode='binary')

    print(test_generator.class_indices)


    print('......... Loading Model .........')
    model = load_model("model.h5")
    #plot_model(model, to_file='model.png', show_shapes = False, show_layer_names = True, rankdir='TB')
    print("Loaded model from disk")

    predictions = model.predict_generator(test_generator)
    print('---------')
    Y_pred = []
    for p in predictions:
        if p < 0.5:
            Y_pred.append(0)
        else:
            Y_pred.append(1)
    
    print("Accuracy:", accuracy_score(Y_pred, test_generator.classes))
    print("Precision:",precision_recall_fscore_support(Y_pred, test_generator.classes)[1][1])
    print("Recall:",precision_recall_fscore_support(Y_pred, test_generator.classes)[0][1])
    print("F_Score:",precision_recall_fscore_support(Y_pred, test_generator.classes)[2][1])
    
    cnf_matrix = confusion_matrix(test_generator.classes, Y_pred)
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
    Plot_Recall_Precision(test_generator.classes, predictions)
    plt.show()

main()