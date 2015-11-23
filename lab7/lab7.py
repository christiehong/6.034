# 6.034 Lab 7 2015: Boosting (Adaboost)

from math import log as ln
INF = float('inf')

# Helper function for pick_best_classifier and adaboost
def fix_roundoff_error(inp, n=15):
    """inp can be a number, a list of numbers, or a dict whose values are numbers.
    * If inp is a number: Rounds the number to the nth decimal digit to reduce
        previous Python roundoff error.  Returns a float.
    * If inp is a list of numbers: Rounds each number as above.  Does not modify
        the original list.
    * If inp is a dictionary whose values are numbers: Rounds each value as
        above.  Does not modify the original dictionary."""
    fix_val = lambda val: round(abs(val),n)*[-1,1][val>=0]
    if isinstance(inp, list): return map(fix_val, inp)
    if isinstance(inp, dict): return {key: fix_val(inp[key]) for key in inp}
    return fix_val(inp)


#### BOOSTING (ADABOOST) #######################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    dic = {}
    weight = 1.0/len(training_points)
    for point in training_points:
        dic[point] = weight

    return dic

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    classifier_to_error_rate = {}
    # for each classifier
    for classifier in classifier_to_misclassified:
        # calculate error rate and append to dictionary
        misclassified_points = classifier_to_misclassified[classifier]
        error_rate = 0
        # add up error of each missclassified point for classifier
        for point in misclassified_points:
            error_rate += point_to_weight[point]
        # append error_rate to dictionary with classifier key
        classifier_to_error_rate[classifier] = error_rate

    return classifier_to_error_rate


def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier.  Best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    
    if use_smallest_error:
        return min(classifier_to_error_rate, key=classifier_to_error_rate.get)

    else:
        # Furthest from 1/2
        for classifier in classifier_to_error_rate:
            classifier_to_error_rate[classifier] = abs(classifier_to_error_rate[classifier] - 0.5)

        classifier_to_error_rate = fix_roundoff_error(classifier_to_error_rate)

        return max(classifier_to_error_rate, key=classifier_to_error_rate.get)



def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    raise NotImplementedError

def is_good_enough(H, training_points, classifier_to_misclassified,
                   mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    raise NotImplementedError

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""
    raise NotImplementedError

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_num_rounds=INF):
    """Performs the Adaboost algorithm for up to max_num_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""
    raise NotImplementedError


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
