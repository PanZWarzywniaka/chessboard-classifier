"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
import scipy.linalg
from scipy import stats

N_DIMENSIONS = 10
PCA_N_DIMENSIONS = 10 # = 40
CHOSEN_FEATURES = []


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray, k: int = None,features: list = None) -> List[str]:
    """Classify a set of feature vectors using a training set.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    """Perform k-nearest neighbour classification."""

    #Use 8-nearest samples to classify, unless other parameter supplied
    if k is None:
        k = 8

    # Use all feature is no feature parameter has been supplied
    if features is None:
        features = np.arange(0, train.shape[1])

    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]

    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance

    k_nearest = np.argsort(-dist, axis=1)[:,:k] #getting arguments of k nearest samples
    labels_guesses = train_labels[k_nearest] #getting nables of recieved arguments

    #using stats.mode to get mode of each labels_guesses for example ['k','k','k','b','b'] will return ['k']
    labels, _ = stats.mode(labels_guesses, axis=1) #skipping unnesecary choices part

    labels = np.ndarray.flatten(labels) #transforming 2-d array (N,1) to vector (N,)
    return labels

def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:

        Already reduced: fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    train_data_reduced = np.array(model["fvectors_train_reduced"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(train_data_reduced, labels_train, fvectors_test)

    return labels

# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.



def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    #TODO: some more advanced technique can be used than PCA, considering train labels as well

    v = model["eigen_vectors"]
    mean = model["train_mean"]

    reduced_data = np.dot((data - mean), v) #to PCA_N_DIMENSIONS

    # features = model["features"]
    # reduced_data = reduced_data[:, features] #to N_DIMENTIONS

    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    model = {}
    model["labels_train"] = labels_train.tolist()
    
    pc_axes = get_eigen_vectors(fvectors_train).tolist()
    model["eigen_vectors"] = pc_axes

    mean = np.mean(fvectors_train).tolist()
    model["train_mean"] = mean

    model["fvectors_train_reduced"] = reduce_dimensions(fvectors_train,model).tolist()
    return model





def get_eigen_vectors(train_data: np.ndarray) -> np.ndarray:
    covx = np.cov(train_data, rowvar=0)
    N = covx.shape[0]
    _, v = scipy.linalg.eigh(covx, eigvals=(N - PCA_N_DIMENSIONS, N - 1))

    v = np.fliplr(v)
    return v


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
   =      list[str]: A list of one-character strings representing the labels for each square.
    """

    return classify_squares(fvectors_test, model)


def feature_selection(train_labels: np.ndarray ,train_data: np.ndarray) -> list:
    return get_sorted_features_div_only(train_labels,train_data)[0:N_DIMENSIONS-1]
     

def get_sorted_features_div_only(train_labels,train_data):
    div_score=get_divergence_score(train_labels,train_data)
    return np.argsort(-div_score)

def get_divergence_score(train_labels,train_data):
    
    divergence_score = np.zeros(train_data.shape[1],)
    classes = np.unique(train_labels)
    nclasses = classes.shape[0]
    
    for i in range(nclasses): #from first label to last one
        for j in range(i+1, nclasses): # start from i+1 to avoid repetition of classes
            class_1 = train_data[train_labels == classes[i], :]
            class_2 = train_data[train_labels == classes[j], :]
            if(class_1.shape[0]>1 and class_2.shape[0]>1): #we only take those pairs for which we have enough data (more than one sample)
                divergence_score += divergence(class_1, class_2)
    
    return divergence_score

def divergence(class1, class2):
    """compute a vector of 1-D divergences
    
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    
    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (
        1.0 / v1 + 1.0 / v2
    )

    return d12