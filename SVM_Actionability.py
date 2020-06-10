from sklearn.datasets import make_classification
from sklearn import preprocessing
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import time
import math
import random
import argparse
import sys


h = .02  # step size in the mesh
RANDOM_STATE = 12
GAMMA=2
DEGREE=2

random.seed(RANDOM_STATE)

# Define the Linear SVM
def create_linear_svm(num_points):
    # Create the simulated linear data and scale it
    X, y = make_classification(n_features=2, n_informative=2, n_redundant=0, random_state=RANDOM_STATE,
                               n_samples=num_points)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X = min_max_scaler.fit_transform(X)

    y = np.asarray(y)
    y[y < 1] = -1

    clf = SVC(kernel='linear', random_state=RANDOM_STATE)
    clf.fit(X, y)
    return X, y, clf.coef_[0], clf

# Find the Lambda Variable
def lambda_var(x_i, w, x_0, v):
    num = 2 * np.matmul(v.T, np.multiply(w, x_0 - x_i))
    denom = np.matmul(v.T, v)
    return -num/denom

# Find the Lagrangian Gradient
def lagrangian_grad(x_i, w, x_0, v):
    term_1 = 2 * np.multiply(w, x_0 - x_i)
    term_2 = lambda_var(x_i, w, x_0, v) * v

    return term_1 + term_2

# Find the Weighted Distance
def distance(x, y, w):
    distance = np.matmul(np.multiply(w, x - y).T, np.multiply(w, x - y))
    return distance

# Find the Unweighted Distance
def unweighted_distance(x, y):
    distance = np.matmul((x - y).T, (x-y))
    return distance

# Find the Gradient Descent Solution
def find_nearest_linear_point(x_0, v, w, x_1, eta=0.15, stopping_distance=0.0001):
    x_i = x_0
    x_i_plus_1 = x_1

    num_iterations = 0
    while unweighted_distance(x_i, x_i_plus_1) > stopping_distance:
        x_i = x_i_plus_1
        x_i_plus_1 = x_i + eta * lagrangian_grad(x_i, w, x_0, v)
        num_iterations += 1

    print("Number of iterations is {}".format(num_iterations))

    return x_i_plus_1, num_iterations

# This function finds the nearest support vector
def find_nearest_sv(x_0, svm, weights, goal):
    sv = []
    nearest_distance = math.inf

    for j in range(0, len(svm.support_vectors_)):
        dist = distance(x_0, svm.support_vectors_[j], weights)
        current_sv = svm.support_vectors_[j].reshape(1, -1)

        if dist < nearest_distance and svm.decision_function(current_sv)[0]/float(goal) > 1:
            nearest_distance = dist
            sv = svm.support_vectors_[j]

    return sv, nearest_distance

# This function compares the Gradient Descent and Nearest Support Vector
def run_linear_algo(x_0, y_0, coefs, weights, svm, graph_examples):
    goal = -y_0

    # Set the default y value if x is 0 for x_1
    x_1_0 = 0

    x_1 = np.asarray([x_1_0, (goal - x_1_0 * coefs[0]) / coefs[1]])

    # Measure the gradient descent point
    start = time.time()
    nearest_grad, num_iter = find_nearest_linear_point(x_0, coefs, weights, x_1)
    grad_run_time = time.time() - start
    grad_dist = distance(x_0, nearest_grad, weights)

    # The below code shows the example
    x = np.linspace(-1, 1, 100)
    # coefs[0]*x + coefs[1]*y = 0
    y = (-coefs[0] * x) / coefs[1]

    y_pos = (1 - coefs[0] * x) / coefs[1]
    y_neg = (-1 - coefs[0] * x) / coefs[1]

    if graph_examples:
        fig, ax = plt.subplots()
        col_0 = 'b' if y_0 == 1 else 'r'
        col_1 = 'r' if y_0 == 1 else 'b'
        ax.plot(x, y, '-k') #, label='{0:.2f}x + {1:.2f}y = 0'.format(coefs[0], coefs[1]))
        ax.plot(x, y_pos, 'b--', label='positive margin')
        ax.plot(x, y_neg, 'r--', label='negative margin')
        ax.plot(x_0[0], x_0[1], col_0 + 'o', label='x_0 label')
        print(x_1)
        ax.plot(x_1[0], x_1[1], 'k^')
        ax.plot(nearest_grad[0], nearest_grad[1], col_1 + 'o', label='gradient descent solution')
        plt.ylim([-1, 1])
        plt.title(
            '2D Action with Weight Vector [{0:.3f}, {1:.3f}], {2:.0f} iterations'.format(weights[0], weights[1], num_iter))
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend()
        plt.grid()
        plt.show()

    # Measure the nearest support vector
    nearest_sv, nearest_sv_distance = find_nearest_sv(x_0, svm, weights, goal)

    return grad_dist, nearest_sv_distance, grad_run_time

# Run the linear SVM algorithm
def run_linear_svm(weights = np.asarray([1, 0.1]), result_dir='', num_points=1000, summary_results=True,
                   graph_examples=False):
    # Create the data and define the weights
    X, y, coefs, svm = create_linear_svm(num_points)

    ana_distances = []
    grad_distances = []
    nearest_sv_distances = []

    grad_times = []

    for i in range(0, len(X)):
        x_0 = X[i, :]
        y_0 = y[i]
        grad_dist, nearest_sv_dist, grad_run_time = \
            run_linear_algo(x_0, y_0, coefs, weights, svm, graph_examples)

        grad_distances = np.append(grad_distances, grad_dist)
        nearest_sv_distances = np.append(nearest_sv_distances, nearest_sv_dist)

        grad_times = np.append(grad_times, grad_run_time)

    # The commented out code below saves information on the linear SVM
    if summary_results:
        print("Statistics on Gradient Descent Prediction Run Time. Average: {}s, Max: {}s, Min: {}s"
              .format(np.mean(grad_times), max(grad_times), min(grad_times)))

        # Create a boxplot of the Distance Data
        distance_data = [grad_distances, nearest_sv_distances]
        plt.boxplot(distance_data)
        plt.title("Normalized Weighted Distance from Solution to Initial Point")
        plt.xticks([1, 2], ['Gradient Descent', 'Nearest Support Vector'])
        plt.savefig(result_dir + 'linear_distances_boxplot.png')
        plt.close()

        t_val, p_val = st.ttest_ind(grad_distances, nearest_sv_distances)
        print("Gradient Descent Average Distance {:.3f}".format(np.mean(grad_distances)))
        print("Nearest SV Average Distance {:.3f}".format(np.mean(nearest_sv_distances)))
        print("One sided p-val of Significance is: {:4f}".format(p_val / 2))
        print(t_val)

# Create nonlinear datasets
def create_nonlinear_data(num_points, type='moon'):
    if type == 'moon':
        X, y = make_moons(noise=0.3, random_state=RANDOM_STATE, n_samples=num_points)
    elif type == 'circle':
        X, y = make_circles(random_state=RANDOM_STATE, n_samples=num_points)
    else:
        raise Exception("Invalid data type.")
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X = min_max_scaler.fit_transform(X)

    y = np.asarray(y)
    y[y < 1] = -1

    return X, y

# Create a nonlinear SVM
def create_nonlinear_SVM(X, y, kernel='rbf'):
    if kernel == 'rbf':
        svm = SVC(kernel='rbf', gamma=GAMMA, C=1)
    elif kernel == 'poly':
        svm = SVC(kernel='poly', degree=DEGREE)

    svm.fit(X, y)
    return svm

# Create the mesh used for graphing
def create_mesh(X, svm):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(svm, "decision_function"):
        Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])

        # Adjust the Z values to create the margins
        Z = np.where(Z <= -1, -1, Z)
        Z = np.where(Z >= 1, 1, Z)
        Z = np.where(abs(Z) < 1, 0, Z)

    else:
        Z = svm.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    return xx, yy, Z

# Calculate RBF Between two points
def calculate_rbf(x_1, x_2, gamma=GAMMA):
    squared_euclidean_distance = 0
    for i in range(0, len(x_1)):
        squared_euclidean_distance += (x_1[i] - x_2[i])**2

    return math.exp(-gamma * squared_euclidean_distance)

# Calculate RBF Dual Vector
def calculate_rbf_dual_vector(x_i, svm, gamma):
    dual_vector = np.zeros(len(x_i))
    for j in range(0, len(svm.support_vectors_)):
        dual_coef = svm.dual_coef_[0][j]
        support_vector = svm.support_vectors_[j]
        dual_vector = np.add(dual_vector, dual_coef *
                             2 * gamma * (x_i - support_vector) * calculate_rbf(x_i, support_vector, gamma))

    return dual_vector

# Calculate RBF Lambda Variable
def rbf_lambda_var(x_i, w, x_0, svm, gamma):
    dual_vector = calculate_rbf_dual_vector(x_i, svm, gamma)
    num = 2 * np.matmul(dual_vector.T, np.multiply(w, x_0 - x_i))
    denom = np.matmul(dual_vector.T, dual_vector)
    return num/denom

# Calculate RBF Gradient
def rbf_lagrangian_grad(x_i, w, x_0, svm, gamma):
    term_1 = - 2 * np.multiply(w, x_0 - x_i)
    lambda_val = rbf_lambda_var(x_i, w, x_0, svm, gamma)
    summation_val = calculate_rbf_dual_vector(x_i, svm, gamma)
    term_2 = lambda_val * summation_val

    return term_1 + term_2

# Calculate Polynomial Dual Vector
def calculate_poly_dual_vector(x_i, svm, d):
    dual_vector = np.zeros(len(x_i))
    for j in range(0, len(svm.support_vectors_)):
        dual_coef = svm.dual_coef_[0][j]
        support_vector = svm.support_vectors_[j]
        dual_vector = np.add(dual_vector, dual_coef * d * (support_vector) *
                             (np.matmul(x_i.T, support_vector))**(d-1))

    return dual_vector

# Calculate Polynomial Lambda Variable
def poly_lambda_var(x_i, w, x_0, svm, degree):
    dual_vector = calculate_poly_dual_vector(x_i, svm, degree)
    num = 2 * np.matmul(dual_vector.T, np.multiply(w, x_0 - x_i))
    denom = np.matmul(dual_vector.T, dual_vector)
    return num/denom

# Calculate Polynomial Gradient
def poly_lagrangian_grad(x_i, w, x_0, svm, degree):
    term_1 = - 2 * np.multiply(w, x_0 - x_i)
    lambda_val = poly_lambda_var(x_i, w, x_0, svm, degree)
    summation_val = calculate_poly_dual_vector(x_i, svm, degree)
    term_2 = lambda_val * summation_val

    return term_1 + term_2

# Find the nearest nonlinear point
def find_nearest_nonlinear_point(x_0, svm, weights, x_1,
                                 gamma=GAMMA, degree=DEGREE, eta=0.15, stopping_distance=0.0001, method='rbf'):
    x_i = x_0
    x_i_plus_1 = x_1

    num_iterations = 0
    while unweighted_distance(x_i, x_i_plus_1) > stopping_distance:
        x_i = x_i_plus_1
        if method == 'rbf':
            x_i_plus_1 = x_i - eta * rbf_lagrangian_grad(x_i, weights, x_0, svm, gamma)
        elif method == 'poly':
            x_i_plus_1 = x_i - eta * poly_lagrangian_grad(x_i, weights, x_0, svm, degree)

        num_iterations += 1

    print("Number of iterations is {}".format(num_iterations))

    return x_i_plus_1, num_iterations

# Run the algorithm for a nonlinear point
def run_nonlinear_algo(x_0, y_0, weights, svm, xx, yy, Z, method='rbf', graph_examples=False):
    goal = -y_0

    # Set a goal point
    nearest_sv, nearest_sv_distance = find_nearest_sv(x_0, svm, weights, goal)
    x_1 = nearest_sv

    # Run the gradient descent algo
    start = time.time()
    nearest_gd, num_iter = find_nearest_nonlinear_point(x_0, svm, weights, x_1, method=method)
    run_time = time.time() - start
    grad_desc_distance = distance(nearest_gd, x_0, weights)

    # The below code will graph the result
    if graph_examples:
        fig, ax = plt.subplots()
        cm_bright = ListedColormap(['#FF0000', '#FFFFFF', '#0000FF'])

        ax.contourf(xx, yy, Z, cmap=cm_bright, alpha=.5)

        plt.xlabel('x_1', color='#1C2833')
        plt.ylabel('x_2', color='#1C2833')


        col_0 = 'b' if y_0 == 1 else 'r'
        col_1 = 'r' if y_0 == 1 else 'b'
        ax.plot(x_0[0], x_0[1], col_0 + 'o', label='x_0 label')
        ax.plot(x_1[0], x_1[1], 'k^', label='x_1')
        ax.plot(nearest_gd[0], nearest_gd[1], col_1 + 'o',
                label='nearest grad descent (iterations: {})'.format(num_iter))
        ax.plot(nearest_sv[0], nearest_sv[1], col_1 + 's',
                label='nearest SV')


        plt.title('2D Action with Weight Vector [{0:.3f}, {1:.3f}], \n Grad Descent Distance - Nearest SV Distance: {2:.0f}'
                  .format(weights[0], weights[1], grad_desc_distance - nearest_sv_distance))
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend()
        plt.grid()
        plt.show()


    return grad_desc_distance, nearest_sv_distance, run_time

# Run a nonlinear algorithm
def run_nonlinear_svm(weights = np.asarray([1, 0.2]), result_dir='', num_points=100, data_type='moon', svm_type='rbf',
                      summary_results=True, graph_examples=False):
    # Define the data and SVM
    X, y = create_nonlinear_data(num_points, type=data_type)

    if svm_type == 'rbf':
        svm = create_nonlinear_SVM(X, y, kernel='rbf')
    elif svm_type == 'poly':
        svm = create_nonlinear_SVM(X, y, kernel='poly')
    else:
        raise Exception('Invalid SVM type.')

    # Create the mesh that will be used for graphing
    xx, yy, Z = create_mesh(X, svm)

    grad_distances = []
    nearest_svm_distances = []
    grad_times = []

    for i in range(0, len(X)):
        x_0 = X[i, :]
        y_0 = y[i]

        goal = -y_0

        if goal == 1:
            grad_dist, svm_dist, grad_time = run_nonlinear_algo(x_0=x_0, y_0=y_0, weights=weights, svm=svm,
                                                                xx=xx, yy=yy, Z=Z, method=svm_type,
                                                                graph_examples=graph_examples)


        else:
            grad_dist, svm_dist, grad_time = run_nonlinear_algo(x_0=x_0, y_0=y_0, weights=weights, svm=svm,
                                                                xx=xx, yy=yy, Z=Z, method=svm_type,
                                                                graph_examples=graph_examples)

        grad_distances.append(grad_dist)
        nearest_svm_distances.append(svm_dist)
        grad_times.append(grad_time)

    # The below code will print information about the gradient descent algo
    if summary_results:
        print("Statistics on Gradient Descent Prediction Run Time. Average: {}s, Max: {}s, Min: {}s"
              .format(np.mean(grad_times), max(grad_times), min(grad_times)))

        # Create a boxplot of the Distance Data
        distance_data = [grad_distances, nearest_svm_distances]
        plt.boxplot(distance_data)
        plt.title("Normalized Weighted Distance from Solution to Initial Point")
        plt.xticks([1, 2], ['Gradient Descent', 'Nearest Support Vector'])
        plt.savefig(result_dir + 'distances_boxplot.png')
        plt.close()

        t_val, p_val = st.ttest_ind(grad_distances, nearest_svm_distances)
        print("Gradient Descent Average Distance {:.3f}".format(np.mean(grad_distances)))
        print("Nearest SV Average Distance {:.3f}".format(np.mean(nearest_svm_distances)))
        print("One sided p-val of Significance is: {:4f}".format(p_val / 2))
        print(t_val)

# Create the atherosclerosis_dataset
def create_atherosclerosis_data(filename, sheetname, x_cols, y_col):
    atherosclerosis_data = pd.read_excel(filename, sheet_name=sheetname)
    atherosclerosis_data.columns = atherosclerosis_data.columns.str.strip()
    atherosclerosis_data['TOBA_CONSO'] = pd.to_numeric(atherosclerosis_data['TOBA_CONSO'], errors='coerce')

    atherosclerosis_data = atherosclerosis_data.loc[atherosclerosis_data['SUPER_GROU'].isin(['N', 'R', 'P'])]
    atherosclerosis_data = atherosclerosis_data.dropna(subset=['GROUP'], how='any')
    filtered_atherosclerosis = atherosclerosis_data.loc[:, x_cols + [y_col, 'GROUP']]

    X = filtered_atherosclerosis.loc[:, x_cols]

    y = filtered_atherosclerosis.loc[:, y_col]
    y = np.asarray(y)
    y[y == 'N'] = -1
    y[y == 'R'] = 1
    y[y == 'P'] = 1

    risk = filtered_atherosclerosis['GROUP']

    return X, y, risk.to_numpy()

# Create the linear model to fit the Atherosclerosis Risk
def create_linear_model(X, risk):
    reg = LinearRegression().fit(X, risk)
    return reg

# Create SVM for Atherosclerosis Data
def create_atherosclerosis_SVM(X, y, kernel='rbf', gamma=GAMMA, C=1, degree=DEGREE):
    y = y.astype('int')
    if kernel == 'rbf':
        svm = SVC(kernel='rbf', gamma=gamma, C=C,
                  probability=True)
    elif kernel == 'poly':
        svm = SVC(kernel='poly', degree=degree,
                  probability=True)

    svm.fit(X, y)
    return svm

# Test the hyperparameters for rbf
def hyper_parameter_rbf(X_scale, y, gamma_values, C_values):
    predictions = []
    labels = []
    best_gammas = []
    best_Cs = []
    # Repeat the process 10 times
    for i in range(0, 10):
        # First define the train and test sets
        data_train, data_test, labels_train, labels_test = train_test_split(X_scale, y, test_size=0.20)

        best_gamma = 0
        best_C = 0
        max_weighted_f1 = -math.inf
        # Run 5 fold CV on all combinations of values
        for gamma in gamma_values:
            for C in C_values:
                kf = KFold(n_splits=5)

                y_pred = []
                y_truth = []

                for train_index, test_index in kf.split(data_train, groups=labels_train):
                    X_train, X_test = data_train[train_index], data_train[test_index]
                    y_train, y_test = labels_train[train_index], labels_train[test_index]

                    # Fill in the training null values
                    temp_df = pd.DataFrame(X_train)
                    X_train = temp_df.fillna(temp_df.mean()).to_numpy()

                    # Fill in the test null values
                    temp_df = pd.DataFrame(X_test)
                    X_test = temp_df.fillna(temp_df.mean()).to_numpy()

                    svm_rbf = create_atherosclerosis_SVM(X_train, y_train, kernel='rbf', gamma=gamma, C=C)
                    predicted = svm_rbf.predict(X_test)

                    y_pred = np.append(y_pred, predicted)
                    y_truth = np.append(y_truth, y_test)

                # Find the report for this combination
                y_pred = np.asarray(y_pred, dtype=float)
                y_truth = np.asarray(y_truth, dtype=float)

                if sum(y_pred == -1) > 0:
                    report = classification_report(y_truth, y_pred, output_dict=True)
                    f1_score = report['weighted avg']['f1-score']

                    # If higher f1 score, replace gamma, C
                    if f1_score > max_weighted_f1:
                        best_gamma = gamma
                        best_C = C
                        max_weighted_f1 = f1_score

        print("Best weighted f1-score for CV on training set {} was {} with gamma value {} and C value {}"
              .format(i, max_weighted_f1, best_gamma, best_C))

        best_gammas.append(best_gamma)
        best_Cs.append(best_C)

        # Fill in the training null values
        temp_df = pd.DataFrame(data_train)
        data_train = temp_df.fillna(temp_df.mean()).to_numpy()

        # Fill in the test null values
        temp_df = pd.DataFrame(data_test)
        data_test = temp_df.fillna(temp_df.mean()).to_numpy()

        # Create the best RBF SVM
        svm_rbf = create_atherosclerosis_SVM(data_train, labels_train, kernel='rbf', gamma=best_gamma, C=best_C)
        y_pred_rbf = svm_rbf.predict(data_test)
        y_pred_rbf = np.asarray(y_pred_rbf, dtype=float)
        labels_test = np.asarray(labels_test, dtype=float)

        predictions = np.append(predictions, y_pred_rbf)
        labels = np.append(labels, labels_test)

    return predictions, labels

# Test the hyperparameters for polynomial
def hyper_parameter_poly(X_scale, y, degrees):
    predictions = []
    labels = []
    best_degrees = []
    # Repeat the process 10 times
    for i in range(0, 10):
        # First define the train and test sets
        data_train, data_test, labels_train, labels_test = train_test_split(X_scale, y, test_size=0.20)

        best_degree = 0
        max_weighted_f1 = -math.inf
        # Run 5 fold CV on all combinations of values
        for degree in degrees:
            kf = KFold(n_splits=5)

            y_pred = []
            y_truth = []

            for train_index, test_index in kf.split(data_train, groups=labels_train):
                X_train, X_test = data_train[train_index], data_train[test_index]
                y_train, y_test = labels_train[train_index], labels_train[test_index]

                # Fill in the training null values
                temp_df = pd.DataFrame(X_train)
                X_train = temp_df.fillna(temp_df.mean()).to_numpy()

                # Fill in the test null values
                temp_df = pd.DataFrame(X_test)
                X_test = temp_df.fillna(temp_df.mean()).to_numpy()

                svm_poly = create_atherosclerosis_SVM(X_train, y_train, kernel='poly', degree=degree)
                predicted = svm_poly.predict(X_test)

                y_pred = np.append(y_pred, predicted)
                y_truth = np.append(y_truth, y_test)

            # Find the report for this combination
            y_pred = np.asarray(y_pred, dtype=float)
            y_truth = np.asarray(y_truth, dtype=float)

            if sum(y_pred == -1) > 0:
                report = classification_report(y_truth, y_pred, output_dict=True)
                f1_score = report['weighted avg']['f1-score']

                # If higher f1 score, replace degree
                if f1_score > max_weighted_f1:
                    best_degree = degree
                    max_weighted_f1 = f1_score

        print("Best weighted f1-score for CV on training set {} was {} with degree value {}"
              .format(i, max_weighted_f1, best_degree))

        best_degrees.append(best_degree)

        # Fill in the training null values
        temp_df = pd.DataFrame(data_train)
        data_train = temp_df.fillna(temp_df.mean()).to_numpy()

        # Fill in the test null values
        temp_df = pd.DataFrame(data_test)
        data_test = temp_df.fillna(temp_df.mean()).to_numpy()

        # Create the best RBF SVM
        svm_poly = create_atherosclerosis_SVM(data_train, labels_train, kernel='poly', degree=best_degree)
        y_pred_poly = svm_poly.predict(data_test)
        y_pred_poly = np.asarray(y_pred_poly, dtype=float)
        labels_test = np.asarray(labels_test, dtype=float)


        predictions = np.append(predictions, y_pred_poly)
        labels = np.append(labels, labels_test)

    return predictions, labels

# Create the heat maps of feature changes
def create_heat_maps(grad_percents, sv_percents, num_cols, num, result_dir):
    patients = []
    for i in range(0, len(grad_percents)):
        patients.append("Patient {}".format(i + 1))

    grad_percents = grad_percents.T
    sv_percents = sv_percents.T

    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    ax1, ax2 = axes

    min_val = min(np.append(sv_percents, grad_percents))
    max_val = max(np.append(sv_percents, grad_percents))

    im1 = ax1.imshow(grad_percents, cmap='coolwarm', vmin=min_val, vmax=max_val)

    # We want to show all ticks...
    ax1.set_xticks(np.arange(len(patients)))
    ax1.set_yticks(np.arange(len(num_cols)))
    # ... and label them with the respective list entries
    ax1.set_xticklabels(patients)
    ax1.set_yticklabels(num_cols)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax1.set_title("Percent Change for Gradient Descent Solution")

    im2 = ax2.imshow(sv_percents, cmap='coolwarm', vmin=min_val, vmax=max_val)

    # We want to show all ticks...
    ax2.set_xticks(np.arange(len(patients)))
    ax2.set_yticks(np.arange(len(num_cols)))
    # ... and label them with the respective list entries
    ax2.set_xticklabels(patients)
    ax2.set_yticklabels(num_cols)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax2.set_title("Percent Change for Nearest Support Vector")

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)

    fig.tight_layout()
    plt.savefig(result_dir + 'rbf_heatmap_change_{}.png'.format(num))
    plt.close()

# Run the algorithm for the atherosclerosis data
def run_atherosclerosis_algo(x_0, y_0, weights, svm, gamma, degree, method='poly'):
    goal = -y_0

    # Find the nearest support vector
    nearest_sv, nearest_sv_distance = find_nearest_sv(x_0, svm, weights, goal)

    # Find the gradient descent solution
    start = time.time()
    nearest_gd, num_iter = \
        find_nearest_nonlinear_point(x_0, svm, weights, nearest_sv, gamma=gamma, degree=degree, method=method)
    run_time = time.time() - start
    grad_desc_distance = distance(nearest_gd, x_0, weights)

    return grad_desc_distance, nearest_sv_distance, nearest_gd, nearest_sv, run_time

# Test the Atherosclerosis Dataset
def run_atherosclerosis_data(weights=[1, 1, .5, .5, .5, .2, .2, .1, .05, .05], result_dir='', svm_type='rbf',
                             summary_results=True):
    num_cols = ['SUBSC', 'TRIC', 'TRIGL', 'SYST', 'DIAST',
                'BMI', 'WEIGHT', 'CHLST', 'ALCO_CONS', 'TOBA_CONSO']
    y_col = 'SUPER_GROU'
    X, y, risk = create_atherosclerosis_data('atherosclerosis.xls', 'athero-rank', num_cols, y_col)

    # Create the imputed data
    imputed_X = X.fillna(X.mean()).to_numpy()
    lm = create_linear_model(imputed_X, risk)

    # Print the regression information below
    if summary_results:
        print("Linear regression of risk summary:")
        regression_info = {'R^2 value': [lm.score(imputed_X, risk)]}
        for i in range(0, len(num_cols)):
            regression_info[num_cols[i]] = [lm.coef_[i]]
        regression_info['Intercept'] = [lm.intercept_]
        regression_df = pd.DataFrame.from_dict(regression_info)
        print(regression_df)

    # Define the weights
    num_weights = weights

    # Scale the data
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_scale = min_max_scaler.fit_transform(X)


    # The below tests the hyperparameters for the models
    """
    test_labels_rbf, predicted_labels_rbf = hyper_parameter_rbf(X_scale, y,
                                                                [0.01, 0.1, 1, 10, 100], [0.01, 0.1, 1, 10, 100])
    print("Full RBF Classification Report")
    print(classification_report(test_labels_rbf, predicted_labels_rbf))

    test_labels_poly, predicted_labels_poly = hyper_parameter_poly(X_scale, y, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    print("Full Poly Classification Report")
    print(classification_report(test_labels_poly, predicted_labels_poly))
    """

    gamma = 1
    C = 10
    degree = 4
    temp_df = pd.DataFrame(X_scale)
    X_scale = temp_df.fillna(temp_df.mean()).to_numpy()
    if svm_type == 'rbf':
        svm = create_atherosclerosis_SVM(X_scale, y, kernel='rbf', gamma=gamma, C=C)
    elif svm_type == 'poly':
        svm = create_atherosclerosis_SVM(X_scale, y, kernel='poly', degree=degree)
    else:
        raise Exception('Invalid SVM type.')

    grad_distances = []
    grad_probs = []
    nearest_sv_distances = []
    nearest_sv_probs = []

    grad_percents = np.empty((0, len(num_cols)))
    sv_percents = np.empty((0, len(num_cols)))

    x_0_risk = []
    nearest_sv_risk = []
    grad_risk = []

    grad_changes = np.empty((0, len(num_cols)))
    sv_changes = np.empty((0, len(num_cols)))

    run_times = []

    for i in range(0, len(X_scale)):
        print(i / float(len(X_scale)))
        x_0 = X_scale[i, :]
        y_0 = y[i]

        # If you want to move to no risk (-1) and prediction is risk
        if y_0 == 1 and svm.predict(x_0.reshape(1, -1))[0] == 1:
            grad_desc_distance, nearest_sv_distance, grad_vec,\
            nearest_sv, run_time = run_atherosclerosis_algo(x_0=x_0, y_0=y_0, weights=num_weights, svm=svm,
                                                            gamma=gamma, degree=degree, method=svm_type)

            run_times = np.append(run_times, run_time)

            grad_distances.append(grad_desc_distance)
            grad_probs.append(svm.predict_proba(grad_vec.reshape(1, -1))[0][0])
            nearest_sv_distances.append(nearest_sv_distance)
            nearest_sv_probs.append(svm.predict_proba(nearest_sv.reshape(1, -1))[0][0])

            x_0_unscaled = min_max_scaler.inverse_transform(x_0.reshape(1, -1))[0]
            grad_unscaled = min_max_scaler.inverse_transform(grad_vec.reshape(1, -1))[0]
            sv_unscaled = min_max_scaler.inverse_transform(nearest_sv.reshape(1, -1))[0]

            grad_changes = np.append(grad_changes, np.array([grad_unscaled - x_0_unscaled]), axis=0)
            sv_changes = np.append(sv_changes, np.array([sv_unscaled - x_0_unscaled]), axis=0)

            with np.errstate(divide='raise'):
                try:
                    grad_percents = np.append(grad_percents,
                                              np.divide([grad_unscaled - x_0_unscaled], x_0_unscaled), axis=0)
                    sv_percents = np.append(sv_percents, np.divide([sv_unscaled - x_0_unscaled], x_0_unscaled), axis=0)

                except FloatingPointError:
                    print("Unable to calculate percent change for patient {}".format(i))

            x_0_risk = np.append(x_0_risk, lm.predict(x_0_unscaled.reshape(1, -1)))
            nearest_sv_risk = np.append(nearest_sv_risk, lm.predict(sv_unscaled.reshape(1, -1)))
            grad_risk = np.append(grad_risk, lm.predict(grad_unscaled.reshape(1, -1)))

    # The below saves information about the atherosclerosis data
    if summary_results:
        # Create random heat maps
        create_heat_maps(grad_percents[:7], sv_percents[:7], num_cols, 1, result_dir)
        create_heat_maps(grad_percents[100:107], sv_percents[100:107], num_cols, 2, result_dir)
        create_heat_maps(grad_percents[200:207], sv_percents[200:207], num_cols, 3, result_dir)

        # Plot risk of probability for nearest support vector
        plt.plot(x_0_risk, nearest_sv_risk, 'ko', markersize=2)
        x = np.linspace(1, 6, 1000)
        decrease_prob = sum(x_0_risk > nearest_sv_risk) / len(x_0_risk)
        plt.plot(x, x, 'k--')
        plt.title('Risk (Probability of Decreased Risk is {:.4f})'.format(decrease_prob))
        plt.xlabel('Risk at Initial Point')
        plt.ylabel('Risk of Nearest SV')
        plt.savefig(result_dir + 'unweighted_nearest_sv_risk.png')
        plt.close()

        # Plot risk of probability for gradient descent solution
        plt.plot(x_0_risk, grad_risk, 'ko', markersize=2)
        plt.plot(x, x, 'k--')
        decrease_prob = sum(x_0_risk > grad_risk) / len(x_0_risk)
        plt.title('Risk (Probability of Decreased Risk is {:.4f})'.format(decrease_prob))
        plt.xlabel('Risk at Initial Point')
        plt.ylabel('Risk of Grad Descent Solution')
        plt.savefig(result_dir + 'unweighted_grad_descent_risk.png')
        plt.close()

        print("Statistics on Prediction Run Time. Average: {}s, Max: {}s, Min: {}s"
              .format(np.mean(run_times), max(run_times), min(run_times)))

        print("Number of Points: {}".format(len(grad_distances)))

        # Plot/save the feature changes, to use interpretability (i.e. amplitudes) change the weight vectors to all 1's
        feature_changes = pd.DataFrame(columns=['Feature', 'avgGradMove', 'avgSVMove', 'varGradMove', 'varSVMove',
                                                'maxValue', 'minValue', 'amplitudeMean', 'amplitudeMedian'])

        for i in range(0, len(num_cols)):
            normalized_amplitude = abs(grad_changes[:, i])/(min_max_scaler.data_max_[i] - min_max_scaler.data_min_[i])
            row = {'Feature': num_cols[i], 'avgGradMove': np.mean(grad_changes[:, i]),
                   'avgSVMove': np.mean(sv_changes[:, i]), 'varGradMove': np.var(grad_changes[:, i]),
                   'varSVMove': np.var(sv_changes[:, i]), 'maxValue': min_max_scaler.data_max_[i],
                   'minValue': min_max_scaler.data_min_[i], 'amplitudeMean': np.mean(normalized_amplitude),
                   'amplitudeMedian': np.median(normalized_amplitude)}

            feature_changes = feature_changes.append(row, ignore_index=True)

            # Create a box plot of the feature changes
            feature_change_data = [grad_changes[:, i], sv_changes[:, i]]
            plt.boxplot(feature_change_data)
            plt.title("Action for Feature {} (Range: {}-{}) from Initial Point to Solution"
                      .format(num_cols[i], min_max_scaler.data_min_[i], min_max_scaler.data_max_[i]))
            plt.xticks([1, 2], ['Gradient Descent Solution', 'Nearest Support Vector'])
            plt.savefig(result_dir + 'feature_changes_{}.png'.format(num_cols[i]))
            plt.close()

        # Create a box plot of the two heaviest feature changes together
        feature_change_data = [grad_changes[:, 0], grad_changes[:, 1], sv_changes[:, 0], sv_changes[:, 1]]
        plt.boxplot(feature_change_data)
        plt.title("Action for Features {}, {} from Initial Point to Solution".format(num_cols[0], num_cols[1]))
        plt.xticks([1, 2, 3, 4], ['GD {}'.format(num_cols[0]),
                                  'GD {}'.format(num_cols[1]),
                                  'Nearest SV {}'.format(num_cols[0]),
                                  'Nearest SV {}'.format(num_cols[1])])
        plt.savefig(result_dir + 'feature_changes_mixed_features.png')
        plt.close()

        feature_changes.to_csv(result_dir + 'atherosclerosis_feature_change.csv')

        print("Average probability of no risk for Grad Descent solution is {}".format(np.mean(grad_probs)))
        print("Average probability of no risk for nearest SV solution is {}".format(np.mean(nearest_sv_probs)))

        t_val, p_val = st.ttest_ind(grad_distances, nearest_sv_distances)
        print("Gradient Descent Average Distance {:.3f}".format(np.mean(grad_distances)))
        print("Nearest SV Average Distance {:.3f}".format(np.mean(nearest_sv_distances)))
        print("One sided p-val of Significance is: {:4f}".format(p_val / 2))
        print(t_val)

        # Create a boxplot of the Distance Data
        distance_data = [grad_distances, nearest_sv_distances]
        plt.boxplot(distance_data)
        plt.title("Normalized Weighted Distance from Solution to Initial Point")
        plt.xticks([1, 2], ['Gradient Descent Solution', 'Nearest Support Vector'])
        plt.savefig(result_dir + 'distances_boxplot.png')
        plt.close()

        # Create a histogram of the distance data
        bins = np.linspace(0, 1, 100)
        plt.hist(grad_distances, bins, alpha=0.5, label='Gradient Descent Solution')
        plt.hist(nearest_sv_distances, bins, alpha=0.5, label='Nearest Support Vector')
        plt.legend()
        plt.savefig(result_dir + 'distances_histogram.png')
        plt.title("Normalized Weighted Distance from Solution to Initial Point")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform SVM Actionability.')
    parser.add_argument('command', metavar='C', choices=['linear', 'non-linear', 'atherosclerosis'],
                        help='the type of SVM action to perform (\'linear\', \'non-linear\', or \'atheroslcerosis\').')
    parser.add_argument('-w', '--weights', nargs='+', type=float,
                        help='the weights (between 0 and 1) for the associated model (length of 2 for '
                             'linear/non-linear, length of 10 for atherosclerosis).', default=None)
    parser.add_argument('-r', '--result_dir',
                        help='the directory to save the results to (default=\'\')', default='')
    parser.add_argument('-n', '--num_points', type=int,
                        help='the number of points to simulate (default: 100)', default=100)
    parser.add_argument('-d', '--data_type', choices=['circle', 'moon'], default='moon',
                        help='the type of data for the non-linear SVM.')
    parser.add_argument('-svm', '--svm_type', choices=['rbf', 'poly'], default='rbf',
                        help='the type of SVM for the non-linear SVM or atherosclerosis SVM.')
    parser.add_argument('-S', '--summary', action='store_true',
                        help='specify whether to save the summary of the results.')
    parser.add_argument('-g', '--graph_examples', action='store_true',
                        help='specify whether to graph each example point.')


    args = parser.parse_args()
    weights = args.weights


    if args.command == 'linear':
        # Run the Linear SVM with gradient descent and analytical solution, there are five optional arguments
        # 1. the weights, 2. the directory for results, 3. the number of points, 4. save the summary of the results,
        # and 5. graph each example point.
        print("Running Linear SVM Model...")

        if weights is None:
            weights = np.asarray([1, 0.1])
        if len(weights) != 2:
            print("Invalid number of weights.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        print(weights)
        run_linear_svm(weights=np.asarray(weights), result_dir=args.result_dir, num_points=args.num_points,
                       summary_results=args.summary, graph_examples=args.graph_examples)

    elif args.command == 'non-linear':
        # Run the Non-Linear SVM with gradient descent, there are seven optional arguments
        # 1. the weights, 2. the directory for results, 3. the number of points,
        # 4. the type of data (moon or circle), 5. the type of svm (rbf or poly), 6. save the summary of the results,
        # and 7. graph each example point.
        print("Running Non-Linear SVM Model...")

        if weights is None:
            weights = np.asarray([1, 0.1])
        if len(weights) != 2:
            print("Invalid number of weights.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        run_nonlinear_svm(weights=np.asarray(weights), result_dir=args.result_dir, num_points=args.num_points,
                          data_type=args.data_type, svm_type=args.svm_type,
                          summary_results=args.summary, graph_examples=args.graph_examples)

    elif args.command == 'atherosclerosis':
        # Run the atherosclerosis testing
        # Run the Atherosclerosis SVM with gradient descent, there are four optional arguments
        # 1. the weights for the features:
        # ['SUBSC', 'TRIC', 'TRIGL', 'SYST', 'DIAST', 'BMI', 'WEIGHT', 'CHLST', 'ALCO_CONS', 'TOBA_CONSO'],
        # 2. the directory for results, 3. the type of svm (rbf or poly), and 4. save the summary of the results.
        print("Running Atherosclerosis SVM Model...")

        if weights is None:
            weights = [1, 1, .5, .5, .5, .2, .2, .1, .05, .05]
        if len(weights) != 10:
            print("Invalid number of weights.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        run_atherosclerosis_data(weights=weights, result_dir=args.result_dir, svm_type=args.svm_type,
                                 summary_results=args.summary)
