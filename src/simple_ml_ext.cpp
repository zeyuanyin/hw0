#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float *get_X_batch(const float *X, size_t batch_start_index, size_t batch_size, size_t feature_size)
{
    float *X_batch = new float[batch_size * feature_size];
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < feature_size; j++)
        {
            X_batch[i * feature_size + j] = X[(i + batch_start_index) * feature_size + j];
        }
    }
    return X_batch;
}

unsigned char *get_y_batch(const unsigned char *y, size_t batch_start_index, size_t batch_size)
{
    unsigned char *y_batch = new unsigned char[batch_size];
    for (size_t i = 0; i < batch_size; i++)
    {
        y_batch[i] = y[i + batch_start_index];
    }
    return y_batch;
}

float *matrix_dot(const float *X, const float *Y, size_t a, size_t b, size_t c)
{
    /**
     * X: a x b
     * Y: b x c
     * return Z: a x c
     */

    float *Z = new float[a * c];
    for (size_t i = 0; i < a; i++)
    {
        for (size_t j = 0; j < c; j++)
        {
            float sum = 0;
            for (size_t k = 0; k < b; k++)
            {
                sum += X[i * b + k] * Y[k * c + j];
            }
            Z[i * c + j] = sum;
        }
    }

    return Z;
}

float *matrix_softmax(const float *X, size_t a, size_t b)
{
    /**
     * X: a x b
     * return Z = softmax(Z)
     */

    float *Z = new float[a * b];
    for (size_t i = 0; i < a; i++)
    {
        float sum = 0;
        for (size_t j = 0; j < b; j++)
        {
            sum += exp(X[i * b + j]);
        }
        for (size_t j = 0; j < b; j++)
        {
            Z[i * b + j] = exp(X[i * b + j]) / sum;
        }
    }

    return Z;
}

float *matrix_transpose(const float *X, size_t a, size_t b)
{
    /**
     * X: a x b
     * return Z = X.T : b x a
     */

    float *Z = new float[a * b];
    for (size_t i = 0; i < a; i++)
    {
        for (size_t j = 0; j < b; j++)
        {
            Z[j * a + i] = X[i * b + j];
        }
    }

    return Z;
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    for (size_t time = 0; time < m / batch; time++)
    {
        size_t batch_start_index = time * batch;
        float *X_batch = get_X_batch(X, batch_start_index, batch, n);
        unsigned char *y_batch = get_y_batch(y, batch_start_index, batch);

        float *H = matrix_dot(X_batch, theta, batch, n, k);
        float *Z = matrix_softmax(H, batch, k);

        // Z-y
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                if (j == y_batch[i])
                {
                    Z[i * k + j] -= 1;
                }
            }
        }
        float *X_T = matrix_transpose(X_batch, batch, n);
        float *G = matrix_dot(X_T, Z, n, batch, k); // n*k

        // update
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                theta[i * k + j] -= lr * G[i * k + j] / (float)batch;
            }
        }
    }

    // std::cout << "hello sys" << std::endl;

    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch)
        {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
