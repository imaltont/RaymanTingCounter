import numpy as np

class NN:
    def __init__(self, architecture, learning_rate=0.1, seed=99):
        #Initialize network
        np.random.seed(seed)
        num_layers = len(architecture)
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.params_values = {}
        for idx, layer in enumerate(architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            self.params_values["".join(['W', str(layer_idx)])] = np.random.randn(layer_output_size, layer_input_size) * 0.1
            self.params_values["".join(['b', str(layer_idx)])] = np.random.randn(layer_output_size, 1) * 0.1
        return
    def inference(self, inp):
        memory= {}
        a_curr = inp

        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            a_prev = a_curr

            activ_function = layer["activation"]
            w_curr = self.params_values["".join(['W', str(layer_idx)])]
            b_curr = self.params_values["".join(['b', str(layer_idx)])]
            a_curr, z_curr = self.single_layer_inference(a_prev, w_curr, b_curr, activ_function)

            memory["".join(['A', str(idx)])] = a_prev
            memory["".join(['Z', str(layer_idx)])] = z_curr
        return a_curr, memory
    def single_layer_inference(self, inp, w, b, activation):
        print(inp)
        x_curr = np.dot(w, inp) + b

        if activation == "relu":
            activation_func = relu
        elif activation == "sigmoid":
            activation_func = sigmoid
        else:
            raise Exception("This activation function does not exist")

        return activation_func(x_curr), x_curr

    def backpropagation(self, y_hat, y, memory):
        grads_values = {}
        m = y.shape[1]
        y = y.reshape(y_hat.shape)

        dA_prev = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))


        for layer_idx_prev, layer in reversed(list(enumerate(self.architecture))):
            layer_idx_curr= layer_idx_prev + 1
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev
            
            A_prev = memory["".join(['A', str(layer_idx_prev)])]
            Z_curr = memory["".join(['A', str(layer_idx_curr)])]
            W_curr = self.params_values["".join(['W', str(layer_idx_curr)])]
            b_curr = self.params_values["".join(['b', str(layer_idx_curr)])]

            dA_prev, dW_curr, db_curr = self.single_backprop(dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values[''.join(["dW", str(layer_idx_curr)])] = dW_curr
            grads_values[''.join(["db", str(layer_idx_curr)])] = db_curr
    
        return grads_values

    def single_backprop(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation):
        m = A_prev.shape[1]

        if activation == "relu":
            activation_func = relu_der
        elif activation == "sigmoid":
            activation_function = sigmoid_der
        else:
            raise Exception('Not supported activation function')

        dZ_curr = activation_func(dA_curr, Z_curr) 
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)


        return dA_prev, dW_curr, db_curr
        
    def update_weights(self, grads_values):
        for layer_idx, layer in enumerate(self.architecture):
            self.params_values["".join(['W', str(layer_idx)])] -= self.learning_rate * grads_values["".join(['dW', str(layer_idx)])]
            self.params_values["".join(['b', str(layer_idx)])] -= self.learning_rate * grads_values["".join(['db', str(layer_idx)])]

    def train(self, X, Y, epochs):
        cost_history = []
        accuracy_history = []

        for i in range(epochs):
            for n in range(len(X)):
                Y_hat, cashe = self.inference(X[n])
                cost = get_cost_value(Y_hat, Y[n])
                cost_history.append(cost)
                accuracy = get_accuracy_value(Y_hat, Y[n])
                accuracy_history.append(accuracy)

                grads_values = self.backpropagation(Y_hat, Y[n], cashe)
                self.update_weights(grads_values)
        return cost_history, accuracy_history
    def save_model(self):
        raise NotImplementedError

#activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_der(dA, x):
    sig = sigmoid(x)
    return dA * sig * (1-sig)

def relu(x):
    return np.maximum(0, x)
def relu_der(dA, x):
    dX = np.array(dA, copy = True)
    dX[x<=0] = 0
    return dX

#Error function
def get_cost_value(y_hat, y):
    m = y_hat.shape[1]
    cost = -1 / m * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))
    return np.squeeze(cost)
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_
def get_accuracy_value(y_hat, y):
    y_hat_ = convert_prob_into_class(y_hat)
    return (y_hat_ == y).all(axis=0).mean()


training_data = [np.array([x,y]) for x in range(2) for y in range(2)]
train_Y = [np.array([0]), np.array([0]), np.array([0]), np.array([1])]
print(training_data, train_Y)
arch = [
        {"input_dim": 2, "output_dim": 1, "activation":"relu"},
        ]
nn = NN(arch)
nn.train(training_data, train_Y, 100)
for x in training_data:
    print(nn.inference(x))
