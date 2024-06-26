# -*- coding: utf-8 -*-
'''

The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

Last modified 2024-05-07 by Anthony Vanderkop.
Hopefully without introducing new bugs.
'''

### LIBRARY IMPORTS HERE ###
import os
import cv2
import keras
import random
import numpy as np
import tensorflow as tf
from imutils import paths
import matplotlib.pyplot as plt
from keras.src.utils import img_to_array

image_dims = (224, 224, 3)


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(11921048, 'Isabell Sophie', 'Hans'), (11220902, 'Kayathri', 'Arumugam'),
            (11477296, 'Nasya Sze Yuen', 'Liew')]


def load_model():
    """
    Load a pre-trained MobileNetV2 model, add a new output layer, and return the model.

    Returns:
        tf.keras.Model: A TensorFlow Keras Model with the loaded MobileNetV2 base model and a new output layer.
    """
    num_classes = 5
    base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3))

    # Task 3: Replace output layer to match number of classes
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model


def freeze_layers(model):
    """
    Freeze all of the layers in the model except the last one.

    Args:
        tf.keras.Model: A TensorFlow Keras Model

    Returns:
        tf.keras.Model: A TensorFlow Keras Model with the base model freezed

    """
    # Get the total number of layers in the model
    num_layers = len(model.layers)

    # Freeze all layers except the last one
    for layer in model.layers[:-1]:
        layer.trainable = False

    return model


def load_data(path):
    """
    Load the data from the specified path and return the data and labels as numpy arrays.

    Args:
        path (str): The path to the data directory.

    Returns:
        tuple: A numpy array containing the data and the labels.
    """
    imagePaths = sorted(list(paths.list_images(path)))
    class_to_int = map_classes_to_int(path)
    random.seed(42)
    random.shuffle(imagePaths)
    combined = []

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)[..., ::-1]  # Convert BGR to RGB
        image = cv2.resize(image, (image_dims[1], image_dims[0]))  # Resize image
        image = img_to_array(image)  # Convert image to array
        label = imagePath.split(os.path.sep)[-2]  # Get label from path
        int_label = class_to_int[label]  # Convert label to integer
        combined.append(np.append(image, int_label))

    combined = np.array(combined)

    return combined


def split_combined_numpy(combined):
    """
    Split the combined array into features (data) and labels, matching the format of data and labels returned by the original function.

    Args:
        combined (numpy.ndarray): The combined array containing both features and labels.

    Returns:
        tuple: A tuple containing features (data) and labels as separate numpy arrays.
    """
    data = []
    labels = []
    image_dims = (224, 224, 3)

    for row in combined:
        image = row[:-1].reshape(image_dims)  # Reshape image data to original shape
        int_label = int(row[-1])
        labels.append(int_label)
        data.append(image)

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    return data, labels


def map_classes_to_int(path):
    """
    Map class names to numbers based on the folder structure in the given path.

    Args:
        path (str): The path to the directory containing class folders.

    Returns:
        dict: A dictionary mapping class names to numbers.
    """
    class_to_int = {}
    classes = sorted(os.listdir(path))
    for i, class_name in enumerate(classes):
        class_to_int[class_name] = i
    return class_to_int


def split_data(X, Y, train_fraction, randomize=False, eval_set=True):
    """
    Split the data into training and testing sets. If eval_set is True, also create
    an evaluation dataset. There should be two outputs if eval_set there should
    be three outputs (train, test, eval), otherwise two outputs (train, test).
    
    Args:
        X (numpy.ndarray): Input features.
        Y (numpy.ndarray): Corresponding labels.
        train_fraction (float): Fraction of data to use for training.
        randomize (bool, optional): Whether to randomly shuffle the data. Defaults to False.
        eval_set (bool, optional): Whether to create an evaluation dataset. Defaults to True.

    Returns:
        tuple: If eval_set is True, returns (train_X, train_Y, test_X, test_Y, eval_X, eval_Y).
               If eval_set is False, returns (train_X, train_Y, test_X, test_Y).
    """
    num_samples = len(X)
    train_samples = int(num_samples * train_fraction)
    test_samples = int(num_samples * ((1 - train_fraction) / 2))

    if randomize:
        indices = np.random.permutation(num_samples)
        X = X[indices]
        Y = Y[indices]

    train_X = X[:train_samples]
    train_Y = Y[:train_samples]
    test_X = X[train_samples:train_samples + test_samples]
    test_Y = Y[train_samples:train_samples + test_samples]

    train = (train_X, train_Y)
    test = (test_X, test_Y)

    if eval_set:
        eval_X = X[train_samples + test_samples:]
        eval_Y = Y[train_samples + test_samples:]
        eval = (eval_X, eval_Y)
        return train, test, eval
    else:
        return train, test


def confusion_matrix(predictions, ground_truth, plot=False, all_classes=None):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the confusion matrix of the classifier's performance.

    Inputs:
        - predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        - ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
        - plot: boolean. If true, create a plot of the confusion matrix with
                either matplotlib or with sklearn.
        - classes: a set of all unique classes that are expected in the dataset.
                   If None is provided we assume all relevant classes are in 
                   the ground_truth instead.
    Outputs:
        - cm: type np.ndarray of shape (c,c) where c is the number of unique  
              classes in the ground_truth
              
              Each row corresponds to a unique class in the ground truth and
              each column to a prediction of a unique class by a classifier
    '''

    raise NotImplementedError
    return cm


def precision(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's precision
    
    Inputs: see confusion_matrix above
    Outputs:
        - precision: type np.ndarray of length c,
                     values are the precision for each class
    '''
    raise NotImplementedError
    return precision


def recall(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's recall
    
    Inputs: see confusion_matrix above
    Outputs:
        - recall: type np.ndarray of length c,
                     values are the recall for each class
    '''
    raise NotImplementedError
    return recall


def f1(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's f1 score
    Inputs:
        - see confusion_matrix above for predictions, ground_truth
    Outputs:
        - f1: type nd.ndarry of length c where c is the number of classes
    '''

    raise NotImplementedError
    return f1


def k_fold_validation(features, ground_truth, classifier, k=2):
    '''
    Inputs:
        - features: np.ndarray of features in the dataset
        - ground_truth: np.ndarray of class values associated with the features
        - fit_func: f
        - classifier: class object with both fit() and predict() methods which
        can be applied to subsets of the features and ground_truth inputs.
        - predict_func: function, calling predict_func(features) should return
        a numpy array of class predictions which can in turn be input to the 
        functions in this script to calculate performance metrics.
        - k: int, number of sub-sets to partition the data into. default is k=2
    Outputs:
        - avg_metrics: np.ndarray of shape (3, c) where c is the number of classes.
        The first row is the average precision for each class over the k
        validation steps. Second row is recall and third row is f1 score.
        - sigma_metrics: np.ndarray, each value is the standard deviation of 
        the performance metrics [precision, recall, f1_score]
    '''

    # split data
    ### YOUR CODE HERE ###

    # go through each partition and use it as a test set.
    # for partition_no in range(k):
    # determine test and train sets
    ### YOUR CODE HERE###

    # fit model to training data and perform predictions on the test set
    # classifier.fit(train_features, train_classes)
    # predictions = classifier.predict(test_features)

    # calculate performance metrics
    ### YOUR CODE HERE###

    # perform statistical analyses on metrics
    ### YOUR CODE HERE###

    raise NotImplementedError
    return avg_metrics, sigma_metrics


##################### MAIN ASSIGNMENT CODE FROM HERE ######################

def transfer_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform standard transfer learning here.

    Args:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)

    Returns:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of class-wise recall, precision, and f1 scores of the
            model on the test_set (list of np.ndarray)

    '''
    learning_rate, momentum, nesterov = parameters
    metrics = ['accuracy']

    # set the optimizer
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov
    )

    # compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=metrics)

    # train the model
    history = model.fit(
        x=train_set[0],
        y=train_set[1],
        validation_data=eval_set,
        epochs=15
    )

    # Task 7: Plot the training and validation errors and accuracies of standard transfer learning
    plot_learning_curves(history, parameters)

    return model, metrics


def accelerated_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform accelerated transfer learning here.

    Args:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)

    Returns:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the
            model on the test_set (list of np.ndarray)

    '''
    # freeze the model base layers
    model = freeze_layers(model)

    learning_rate, momentum, nesterov = parameters
    metrics = ['accuracy']

    # set the optimizer
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov
    )

    # compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=metrics)

    # train the model
    history = model.fit(
        x=train_set[0],
        y=train_set[1],
        validation_data=eval_set,
        epochs=30
    )

    plot_learning_curves(history, parameters)

    return model, metrics


def plot_learning_curves(history, parameters):
    """
    A function that plots the learning curves of a model based on the training history.

    Args:
        - history: the training history of the model
    """
    learning_rate, momentum, nesterov = parameters

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy (learning rate: ' + str(learning_rate) + ' momentum: ' + str(momentum) + ')')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy_lr' + str(learning_rate) + '_mo' + str(momentum) + '.png')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss (learning rate: ' + str(learning_rate) + ' momentum: ' + str(momentum) + ')')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('loss_lr' + str(learning_rate) + '_mo' + str(momentum) + '.png')
    plt.show()


if __name__ == "__main__":
    # Task 1: Use the small_flower_datatset
    path = 'small_flower_dataset'

    # Task 2: Download a pre-trained MobileNetV2 model
    model = load_model()

    # Task 4: Prepare the data for standard transfer learning
    data, labels = split_combined_numpy(load_data(path))
    train, test, eval = split_data(data, labels, 0.8, True, True)

    # Task 5: Compile and train model with SGD optimizer
    model, metrics = transfer_learning(train, test, eval, model, (0.01, 0.0, False))

    # Task 8: Experiment with 3 different orders of magnitude for the learning rate.
    # model_small_lr, metrics_small_lr = transfer_learning(train, test, eval, model, (0.001, 0.0, False))
    # model_medium_lr, metrics_medium_lr = transfer_learning(train, test, eval, model, (0.1, 0.0, False))
    # model_large_lr, metrics_large_lr = transfer_learning(train, test, eval, model, (1, 0.0, False))

    # Task 14:
    # model, metrics = accelerated_learning(train, test, eval, model, (0.01, 0.0, False))

#########################  CODE GRAVEYARD  #############################
