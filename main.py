import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#load data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df['class'] = LabelEncoder().fit_transform(df['class'].values.ravel())
    features = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    labels = pd.get_dummies(df['class'])
    return train_test_split(features, labels, test_size=0.2, random_state=42)


#initial parameter
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def forward_propagation(X, parameters):
    w1, b1, w2, b2 = parameters['w1'], parameters['b1'], parameters['w2'], parameters['b2']
    z1 = np.dot(w1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache


def cost_funtion(a2, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = -np.sum(logprobs) / m
    return cost


def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]
    w2 = parameters['w2']
    a1, a2 = cache['a1'], cache['a2']
    dz2 = a2 - Y
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
    return grads


def update_parameters(parameters, grads, learning_rate=0.4):
    w1, b1, w2, b2 = parameters['w1'], parameters['b1'], parameters['w2'], parameters['b2']
    dw1, db1, dw2, db2 = grads['dw1'], grads['db1'], grads['dw2'], grads['db2']
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

#model
def model(X, Y, n_h, num_iterations=5000, print_cost=False):
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    parameters = initialize_parameters(n_x, n_h, n_y)


    costs = []
    accuracies = []
    # gradient descent
    for i in range(num_iterations):
        a2, cache = forward_propagation(X, parameters)
        cost = cost_funtion(a2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        predictions = (a2 > 0.5).astype(int)
        accuracy = np.mean(predictions == Y) * 100


        costs.append(cost)
        accuracies.append(accuracy)

        if print_cost and i % 1000 == 0:
            print(f'Iteration： {i}  ，loss：{cost} ， accuracy：{accuracy}%')

    return parameters, costs, accuracies


def predict(parameters, X, Y):
    a2, _ = forward_propagation(X, parameters)
    predictions = (a2 > 0.5).astype(int)
    accuracy = np.mean(predictions == Y) * 100
    # print(f'accuracy：{accuracy}%')
    return predictions, accuracy



def user_input_prediction(parameters, label_encoder):
    user_input = np.array([[float(input(f"please enter {feature} in cm: ")) for feature in ["Sepal length", "Sepal width", "Petal length", "Petal width"]]]).T
    prediction, _ = forward_propagation(user_input, parameters)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction, axis=0).ravel()])
    return predicted_class


def main():
    # load data
    X_train, X_test, y_train, y_test = load_and_prepare_data("Irisdata.txt")

    parameters, costs, accuracies = model(X_train.T.values, y_train.T.values, n_h=4, num_iterations=5000,
                                          print_cost=True)


    predictions, accuracy = predict(parameters, X_test.T.values, y_test.T.values)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(costs)
    plt.title('Loss vs Iterations')
    plt.xlabel('iteration')
    plt.ylabel('loss')


    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Accuracy vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('accuracy（%）')

    plt.show()

    # print prediction and actual result
    print("predict result:\n", predictions)
    print("actual result:\n", y_test.values)

    # user input
    label_encoder = LabelEncoder().fit(pd.read_csv("Irisdata.txt", header=None)[4])
    continue_query = 'y'
    while continue_query.lower() == 'y':
        predicted_class = user_input_prediction(parameters, label_encoder)
        print("The Iris class is:", predicted_class[0])
        continue_query = input("Do you want to continute query？(y/n): ")

    print("Query done")

if __name__ == "__main__":
    main()
