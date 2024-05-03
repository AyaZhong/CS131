# Neural Network Iris Species Predictor#

## Usage ##
1. Run the `main.py` file in your preferred Python IDE or terminal.
Make sure you install pandas、numpy、sklearn、matplotlib before you run it.
Use pandas to process the data, sklearn's train_test_split to segment the dataset,
and LabelEncoder to transform the category labels.

2. Enter the data for prediction when prompted in the output console.
The program will ask for the following inputs:

#Example#:
Enter the sepal length in cm: 5.1
Enter the sepal width in cm: 3.5
Enter the petal length in cm: 1.4
Enter the petal width in cm: 0.2

3. Then the program will output the predicted species based on the input data.


#Model detail#
First, the dataset is divided into a training set and a test set, and the weights of the neural network are initialized.
Then, forward propagation is performed to calculate the output of each layer of the neural network, including linear transformation and activation functions.
Next, we use the loss function to calculate the loss between the predicted values and the actual values. After that, we use backpropagation to calculate the gradient of the loss function with respect to each parameter.
inally, we update the parameters to reduce the value of the loss function.
Then build the model and evaluate the model performance by calculating the accuracy during the training process.


