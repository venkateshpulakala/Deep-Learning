import numpy as np

def softmax(scores):

    exp_scores = np.exp(scores - np.max(scores))

    probabilites = exp_scores / np.sum(exp_scores)

    return probabilites


def cross_entropy(probabilities, correct_class):

    correct_probability = probabilities[correct_class]

    loss = -np.log(correct_probability + 1e-12)

    return loss