# 6.034 Lab 6 2015: Neural Nets & SVMs

from nn_problems import *
from svm_problems import *
from math import e

#### NEURAL NETS ###############################################################

# Wiring a neural net

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

# Optional problem; change TEST_NN_GRID to True to test locally
TEST_NN_GRID = True
nn_grid = [4,2,1]

# Helper functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x >= threshold:
        return 1
    else:
        return 0

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    from math import e
    return 1.0/(1.0 + e**(-steepness*(x-midpoint)))

def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5*(desired_output - actual_output)**2

# Forward propagation
def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    raise NotImplementedError

# Backward propagation
def calculate_deltas(net, input_values, desired_output):
    """Computes the update coefficient (delta_B) for each neuron in the
    neural net.  Uses sigmoid function to compute output.  Returns a dictionary
    mapping neuron names to update coefficient (delta_B values)."""
    raise NotImplementedError

def update_weights(net, input_values, desired_output, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses
    sigmoid function to compute output.  Returns the modified neural net, with
    updated weights."""
    raise NotImplementedError

def back_prop(net, input_values, desired_output, r=1, accuracy_threshold=-.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses sigmoid
    function to compute output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    raise NotImplementedError


#### SUPPORT VECTOR MACHINES ###################################################

# Vector math
def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    raise NotImplementedError

def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    raise NotImplementedError

# Equation 1
def positiveness(svm, point):
    "Computes the expression (w dot x + b) for the given point"
    raise NotImplementedError

def classify(svm, point):
    """Uses given SVM to classify a Point.  Assumes that point's classification
    is unknown.  Returns +1 or -1, or 0 if point is on boundary"""
    raise NotImplementedError

# Equation 2
def margin_width(svm):
    "Calculate margin width based on current boundary."
    raise NotImplementedError

# Equation 3
def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    raise NotImplementedError

# Equations 4, 5
def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""
    raise NotImplementedError

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False.  Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    raise NotImplementedError

# Classification accuracy
def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    raise NotImplementedError


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
