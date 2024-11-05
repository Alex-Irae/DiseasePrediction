import numpy as np
import pickle
import os

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    sig = sigmoid(x)
    return sig + x * sig * (1 - sig)


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class CustomNeuralNetwork:
    def __init__(self,inp=132,out=41,h1=192,h2=192,h3=64,init = None):
        
        self.mode = init
        if init ==None :
            self.mode = 'None'
        

        self.hidden_size_1 = h1
        self.hidden_size_2 = h2
        self.hidden_size_3 = h3
        input_size = inp
        output_size = out 

            
        if init == 'he':
            self.W1 = np.random.randn(input_size, self.hidden_size_1) * np.sqrt(2. / input_size)
            
            self.W2 = np.random.randn(self.hidden_size_1, self.hidden_size_2)* np.sqrt(2. / self.hidden_size_1)
           
            self.W3 = np.random.randn(self.hidden_size_2, self.hidden_size_3) * np.sqrt(2. / self.hidden_size_2)
             
            self.W_output = np.random.randn(self.hidden_size_3, output_size) * np.sqrt(2. / self.hidden_size_3)
              
        elif init == 'xavier':
            limit = np.sqrt(6 / (input_size + self.hidden_size_1))
            self.W1 = np.random.uniform(-limit, limit, (input_size, self.hidden_size_1))
            
            limit = np.sqrt(6 / (self.hidden_size_1 + self.hidden_size_2))
            self.W2 = np.random.uniform(-limit, limit, (self.hidden_size_1, self.hidden_size_2))
            
            limit = np.sqrt(6 / (self.hidden_size_2 + self.hidden_size_3))
            self.W3 = np.random.uniform(-limit, limit, (self.hidden_size_2, self.hidden_size_3))
            
            limit = np.sqrt(6 / (self.hidden_size_3 +output_size))
            self.W_output = np.random.uniform(-limit, limit, (self.hidden_size_3, output_size))
            
            
        else : 
            self.W1 = np.random.randn(input_size, self.hidden_size_1) 
            self.W2 = np.random.randn(self.hidden_size_1, self.hidden_size_2) 
            self.W3 = np.random.randn(self.hidden_size_2, self.hidden_size_3) 
            self.W_output = np.random.randn(self.hidden_size_3, output_size) 
            
        self.b1 = np.zeros((1, self.hidden_size_1))  
        self.b2 = np.zeros((1, self.hidden_size_2))  
        self.b3 = np.zeros((1, self.hidden_size_3)) 
        self.b_output = np.zeros((1, output_size))
        
        
        
    def forward(self, X,training):
        # Forward pass through the network
        
        # First hidden layer (using sigm activation)
        self.Z1 = np.dot(a=X, b=self.W1) + self.b1  
        self.A1 = sigmoid(x=self.Z1)  
        if training:
                self.A1 = self.dropout(self.A1, self.dropout_rate)
        # Second hidden layer (using Relu activation)
        self.Z2 = np.dot(a=self.A1, b=self.W2) + self.b2  
        self.A2 = relu(x=self.Z2)  
        if training:
                self.A2 = self.dropout(self.A2, self.dropout_rate)
        # Third hidden layer (using swish activation)
        self.Z3 = np.dot(a=self.A2, b=self.W3) + self.b3 
        self.A3 = swish(x=self.Z3) 
        if training:
                self.A3 = self.dropout(self.A3, self.dropout_rate)
        # Output layer (using softmax activation for multi-class classification)
        self.Z_output = np.dot(a=self.A3, b=self.W_output) + self.b_output  
        self.A_output = softmax(x=self.Z_output) 
                
        return self.A_output

    def dropout(self, A, rate):
        """
        Apply dropout to the activation matrix A.
        
        Parameters:
        - A : ndarray, activation matrix
        - rate : float, dropout rate

        Returns:
        - A_dropped: ndarray, activation matrix after applying dropout
        """
        keep_prob = 1 - rate
        mask = np.random.rand(*A.shape) < keep_prob
        A_dropped = A * mask
        A_dropped /= keep_prob
        return A_dropped
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute the combined loss.
        
        Parameters:
        - y_pred: ndarray, predicted probabilities (output of softmax).
        - y_true: ndarray, true one-hot encoded labels.

        Returns:
        - loss: float, computed loss value.
        """
        y_true = y_true.toarray() if hasattr(y_true, 'toarray') else y_true

        assert y_pred.shape == y_true.shape, f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"

        m = y_true.shape[0]
        cross_entropy_loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m

        diff = y_pred - y_true
        mse_loss = np.mean(np.square(diff)) 

        loss = cross_entropy_loss + 0.5 * mse_loss  

        return cross_entropy_loss

    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]  
        
        # Output layer gradients
        dZ_output = y_pred - y_true 
        # assert dZ_output.shape == (m, self.W_output.shape[1]), f"Shape mismatch: {dZ_output.shape} vs {self.W_output.shape[1]}"
        
        dW_output = 1 / m * np.dot(np.asarray(self.A3).T, dZ_output)  
        db_output = 1 / m * np.sum(np.asarray(dZ_output), axis=0, keepdims=True)

        # Third hidden layer gradients
        dA3 = np.dot(dZ_output, np.asarray(self.W_output).T)  
        # assert dA3.shape == np.asarray(self.A3).shape, f"Shape mismatch: {dA3.shape} vs {self.A3.shape}"

        dZ3 = np.multiply(dA3, sigmoid_derivative(np.asarray(self.Z3)))  
        dW3 = 1 / m * np.dot(np.asarray(self.A2).T, dZ3)  
        db3 = 1 / m * np.sum(np.asarray(dZ3), axis=0, keepdims=True) 
        if self.regularization == 'l2':
                dW3 += (m/self._lambda) * self.W3
        else :  
            dW3 += self._lambda * np.sign(self.W3)
            
        # Second hidden layer gradients
        dA2 = np.dot(dZ3, np.asarray(self.W3).T)  
        assert dA2.shape == np.asarray(self.A2).shape, f"Shape mismatch: {dA2.shape} vs {self.A2.shape}"
        
        dZ2 = np.multiply(dA2 ,relu_derivative(np.asarray(self.Z2)))  
        dW2 = 1 / m * np.dot(np.asarray(self.A1).T, dZ2)  
        if self.regularization == 'l2':
                dW2 += self._lambda/m * self.W2
        else :  
            dW2 += self._lambda * np.sign(self.W2)
        db2 = 1 / m * np.sum(np.asarray(dZ2), axis=0, keepdims=True) 
        

        dA1 = np.dot(dZ2, np.asarray(self.W2).T) 
        assert dA1.shape == np.asarray(self.A1).shape, f"Shape mismatch: {dA1.shape} vs {self.A1.shape}"
        
        dZ1 = np.multiply(dA1,swish_derivative(np.asarray(self.Z1)))  
        dW1 = 1 / m * np.dot(np.asarray(X).T, dZ1)  
        db1 = 1 / m * np.sum(np.asarray(dZ1), axis=0, keepdims=True)  
        if self.regularization == 'l2':
                dW1 += self._lambda/m * self.W1
        else :  
            dW1 += self._lambda * np.sign(self.W1)

        self.W1 -= learning_rate * dW1  
        self.b1 -= learning_rate * db1  
        self.W2 -= learning_rate * dW2  
        self.b2 -= learning_rate * db2  
        self.W3 -= learning_rate * dW3  
        self.b3 -= learning_rate * db3 
        self.W_output -= learning_rate * dW_output  
        self.b_output -= learning_rate * db_output  
           
        
        
    def train(self, X_train, y_train, epochs=1000, learning_rates=0.01,dropout_rate=0.1,_lambda=1,regularization ='l2'):   

        global learning_rate
        learning_rate = learning_rates 
        self.dropout_rate = dropout_rate
        self._lambda = _lambda
        self.regularization = regularization.strip().lower()
        
        
        for epoch in range(epochs):
            y_pred = self.forward(X=X_train,training=True) 
            loss = self.compute_loss(y_pred=y_pred, y_true=y_train) 
            
            self.backward(X=X_train, y_true=y_train, y_pred=y_pred)

            if epoch % 100 == 0: 
                print(f'Epoch {epoch}, Loss: {loss}')
        
        model = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'W_output': self.W_output, 'b_output': self.b_output,
        }
        save_model(model, f'cnn-{self.mode}-{epochs}-{learning_rate}-{self.hidden_size_1}-{self.hidden_size_2}-{self.hidden_size_3}')

    def predict(self, X):
        """
        Predict the class probabilities for the input samples X.
        Returns:
        - probabilities: A 2D NumPy array of shape (n_samples, n_classes) containing class probabilities.
        """
        y_pred = self.forward(X=X,training=False)  
        return y_pred  




def save_model(model, model_name):
    """
    Saves the trained model to a file using pickle.
    
    Parameters:
    - model: Trained model to save.
    - model_name: str, name to use for the saved model file.
    """
    
    models_directory = os.path.join(os.getcwd(), 'models')

    os.makedirs(models_directory, exist_ok=True)

    model_path = os.path.join(models_directory, f'{model_name}.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
        
        
