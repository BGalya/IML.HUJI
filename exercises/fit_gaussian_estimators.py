from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import pandas as pd
import plotly as plotly
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    X = np.random.normal(mu, sigma, size=1000)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(X)
    print((univariate_gaussian.mu_, univariate_gaussian.var_))

    # Question 2 - Empirically showing sample mean is consistent
    fig_q2 = go.Figure()
    fig_q2.add_trace(go.Scatter(x=[i for i in range(10, 1000, 10)],
                                y=[abs(np.mean(X[:i]) - mu) for i in range(10, 1000, 10)], mode='markers+lines',
                                marker=dict(color="red")))

    fig_q2.update_layout(title="<b>Absolute error of estimation, as a function of the sample size</b>")
    fig_q2.update_xaxes(title_text="Absolute distance between the estimated- and true value of the expectation")
    fig_q2.update_yaxes(title_text="Sample size")
    fig_q2.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = univariate_gaussian.pdf(X)
    fig_q3 = go.Figure()
    fig_q3.add_trace(go.Scatter(x=X, y=pdfs, mode='markers'))

    fig_q3.update_layout(title="<b>Empirical PDF, as a function of sample values</b>")
    fig_q3.update_xaxes(title_text="Sample values")
    fig_q3.update_yaxes(title_text="PDF")
    fig_q3.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)
    multivariate_gaussian = MultivariateGaussian()
    estimator = multivariate_gaussian.fit(X)
    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    axis = np.linspace(-10, 10, 200)
    z_axis = []
    for f3 in axis:
        f3_vals = []
        for f1 in axis:
            vals = multivariate_gaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov, X)
            f3_vals.append(vals)
        z_axis.append(f3_vals)

    fig_q5 = go.Figure(data=go.Heatmap(x=axis, y=axis, z=np.array(z_axis)))
    fig_q5.update_layout(title="<b>Heatmap of Log Likelihood Evaluation</b>")
    fig_q5.update_xaxes(title_text="f1")
    fig_q5.update_yaxes(title_text="f3")
    fig_q5.show()

    # Question 6 - Maximum likelihood
    max_index = np.argmax(z_axis)
    max_indices = np.unravel_index(max_index, shape=np.array(z_axis).shape)
    print(format(axis[max_indices[1]], ".3f"), format(axis[max_indices[0]], ".3f"))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
