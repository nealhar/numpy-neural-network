import numpy as np

np.random.seed(0)


# generate a 2D dataset shaped like spirals
def spiral_data(samples, classes):
    # labeled data for input
    X = np.zeros((samples * classes, 2))
    # the correct answers to compare predictions against
    y = np.zeros(samples * classes, dtype=np.uint8)
    
    # loops over each class and fills X and y, each spiral arm is one class
    for class_number in range(classes):
        # finds index range that belongs to this class
        ix = range(samples * class_number, samples * (class_number + 1))
        # r is distance from center
        r = np.linspace(0.0, 1, samples)
        # is different angle offsets, with random noise added
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.2
        # stacks the coordinates to produce the actual 2D spiral pattern
        X[ix] = np.column_stack([r * np.sin(t * 2.5), r * np.cos(t * 2.5)])
        # sets labels to the points
        y[ix] = class_number
    # outputs dataset
    return X, y


# Layers class
class Layer_Dense:
    """
    Fully-connected (dense) layer:
      output = inputs·weights + biases

    Backprop computes:
      dweights = inputs^T · dvalues
      dbiases  = sum(dvalues over batch)
      dinputs  = dvalues · weights^T
    """
    # constructor to set up the layers shape
    def __init__(self, n_inputs, n_neurons):
        # start with small weights and no bias
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons), dtype=np.float32)

    # forward pass function, produces layer output for the next layer
    def forward(self, inputs):
        # saves inputs for backprop
        self.inputs = inputs
        # regression - matrix multiply and add bias
        self.output = np.dot(inputs, self.weights) + self.biases

    # backpropogation function
    def backward(self, dvalues):
        # computes partial to know how to change weights
        self.dweights = np.dot(self.inputs.T, dvalues)
        # computes partials to know how to change bias
        self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)

        # sends gradients to previous layer so earlier weights can learn
        self.dinputs  = np.dot(dvalues, self.weights.T)


# Uses ReLU activation function to allow for nonlinearities
class Activation_ReLU:
    """
    ReLU activation:
      output = max(0, inputs)

    Backprop:
      passes gradient through where input > 0, else 0
    """
    # forward pass
    def forward(self, inputs):
        # if negative send 0, if positive use the input
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    # if input was less than 0 then set gradient to 0
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# defines softmax for output layer
class Activation_Softmax:
    # defines forward pass and applies the softmax formula
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


# Loss base class
class Loss:
    # calls subclass to get per-sample losses and averages them
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        return np.mean(sample_losses)


# categorical cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # sparse labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[np.arange(samples), y_true]

        # one-hot labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # return -log loss
        return -np.log(correct_confidences)

# provides simplified backward
class Softmax_CategoricalCrossentropy(Loss):
    """
    Combined Softmax + Categorical Cross-Entropy.

    Key trick:
      If softmax outputs are p and labels are y (one-hot),
      gradient wrt logits z is:
        dL/dz = (p - y) / batch_size

    This is simpler + more numerically stable than doing
    softmax backward + cross-entropy backward separately.
    """
    # creates a softmax object with a categorical cross entropy inside
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # produces probabilities and computes mean loss
    def forward(self, inputs, y_true):
        # softmax forward
        self.activation.forward(inputs)
        self.output = self.activation.output
        # loss forward
        return self.loss.calculate(self.output, y_true)

    # backward to normalize gradients
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # if one-hot, convert to class indices
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # copy probabilities and subtract 1 at the correct class index
        self.dinputs = dvalues.copy()
        self.dinputs[np.arange(samples), y_true] -= 1

        # average over batch
        self.dinputs = self.dinputs / samples


# stochastic gradient descent optimizer
class Optimizer_SGD:
    """
    SGD optimizer with optional momentum and learning rate decay.

    Without momentum:
      w <- w - lr * dw

    With momentum:
      v <- momentum * v - lr * dw
      w <- w + v
    
    With decay:
      lr_current = lr_initial / (1 + decay * iterations)
    """
    # stores hyperparameters and a counter
    def __init__(self, learning_rate=0.1, momentum=0.0, decay=0.0):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        # smoothing/acceleration
        self.momentum = momentum
        # gradually reduces learning rate
        self.decay = decay
        self.iterations = 0

    # updates one layers weights/biases
    def update_params(self, layer):
        # if decay is enabled then compute decayed learning rate
        if self.decay:
            current_lr = self.initial_lr * (1.0 / (1.0 + self.decay * self.iterations))
        else:
            current_lr = self.learning_rate
        # chooses momentum vs vanilla SGB path
        if self.momentum:
            # initialize velocity buffers the first time we see this layer
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)

            # update velocities
            layer.weight_momentums = self.momentum * layer.weight_momentums - current_lr * layer.dweights
            layer.bias_momentums   = self.momentum * layer.bias_momentums   - current_lr * layer.dbiases

            # apply updates
            layer.weights += layer.weight_momentums
            layer.biases  += layer.bias_momentums
        else:
            # vanilla SGD
            layer.weights -= current_lr * layer.dweights
            layer.biases  -= current_lr * layer.dbiases

        self.iterations += 1

# generate the data
X, y = spiral_data(samples=100, classes=3)

# build the model, 2->256->256->3
dense1 = Layer_Dense(2, 256)
relu1  = Activation_ReLU()
dense2 = Layer_Dense(256, 256)
relu2  = Activation_ReLU()
dense3 = Layer_Dense(256, 3)

# computes loss and gives correct gradients 
loss_activation = Softmax_CategoricalCrossentropy()

# sets training hyperparameters
optimizer = Optimizer_SGD(learning_rate=0.1, momentum=0.95, decay=1e-4)

# number of passes through dataset
epochs = 10000

for epoch in range(epochs + 1):
    # forward pass
    dense1.forward(X)
    relu1.forward(dense1.output)
    dense2.forward(relu1.output)
    relu2.forward(dense2.output)
    dense3.forward(relu2.output)

    # compute loss
    loss = loss_activation.forward(dense3.output, y)

    # accuracy: compare predicted class vs true class
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    # backward pass
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    relu2.backward(dense3.dinputs)
    dense2.backward(relu2.dinputs)
    relu1.backward(dense2.dinputs)
    dense1.backward(relu1.dinputs)

    # update weights/biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)

    # print progress
    if epoch % 500 == 0:
        print(f"epoch: {epoch}, loss: {loss:.4f}, acc: {accuracy:.3f}")

print("\nFinal accuracy: {:.1f}%".format(accuracy * 100))