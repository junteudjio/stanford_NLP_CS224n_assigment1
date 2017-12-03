import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE

        # we first remove the max for each row before computing the softmax
        # we do this for numericall stability in case of large values
        # and we are allowed to because softmax is invariant to translations
        per_row_max = np.max(x, axis=1).reshape(-1, 1)
        x = x - per_row_max

        x = np.exp(x)
        per_row_sum = np.sum(x, axis=1).reshape(-1, 1)
        x = 1. * x / per_row_sum  # we multiply be 1. to ensure float division

        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE

        # this  means that x's shape is something like (34,)
        # what we do is to reshape it as (1,34) matrix and perfom the exact same steps above
        # and reshape it back to its original shape
        x = x.reshape(1, -1)

        # exact same steps as above
        # NOTE: we can't factorized code since the assignments requires to write code in specific area
        per_row_max = np.max(x, axis=1).reshape(-1, 1)
        x = x - per_row_max
        x = np.exp(x)
        per_row_sum = np.sum(x, axis=1).reshape(-1, 1)
        x = 1. * x / per_row_sum  # we multiply be 1. to ensure float division

        # we reshape x to its original shape
        x = x.reshape(orig_shape)

        ### END YOUR CODE

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    test1 = softmax(np.array([0, 0, 0, 0]))
    ans1 = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[0, 0, 0, 0]]))
    ans2 = np.array([[0.25, 0.25, 0.25, 0.25]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[0]]))
    ans3 = np.array([[1.]])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)
    ### END YOUR CODE


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
