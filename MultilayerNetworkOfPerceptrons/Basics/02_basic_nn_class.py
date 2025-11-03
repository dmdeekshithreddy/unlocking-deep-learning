import numpy as np
import numpy.typing as npt

class neuralNetwork:
    # Initialize the network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate) -> None:
        # set the number of nodes in each layer
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.weights_ih = np.random.normal(loc=0.0, 
                                           scale=pow(self.hidden_nodes, -0.5), 
                                           size=(self.input_nodes, self.hidden_nodes))
        self.weights_ho = np.random.normal(loc=0.0, 
                                           scale=pow(self.output_nodes, -0.5), 
                                           size=(self.hidden_nodes, self.output_nodes))
        


        # activation function is the sigmoid function
        # Since activation function is specific to the object
        # it should be defined as an instance method
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, list_input, list_target):
        # convert the input and target array into numpy array and tranpose
        nparr_input = np.transpose(np.array(object=list_input))
        nparr_target = np.transpose(np.array(object=list_target))

        # calculate the combined inputs feed into the hidden layer
        hidden_inputs = np.dot(a=self.weights_ih, b=nparr_input)
        # calculate the outputs from the hidden layer
        hidden_outputs = self.activation_function(x=hidden_inputs)
        # calculate the combined inputs feed into the final layer
        final_inputs = np.dot(a=self.weights_ho, b=hidden_outputs)
        # calculate the outputs from the final layer
        final_outputs = self.activation_function(x=final_inputs)

        # calculate the error
        output_errors = nparr_target - final_outputs

        # calculate hidden layer error
        hidden_errors = np.dot(np.transpose(self.weights_ho), output_errors)

        # 

    def query(self, list_input):
        # convert inputs list to 2D array
        nparr_input = np.transpose(np.array(object=list_input))

        # calculate the combined inputs feed into hidden layer
        hidden_inputs = np.dot(a=self.weights_ih, b=nparr_input)
        # calculate the outputs from the hidden layer
        hidden_outputs = self.activation_function(x=hidden_inputs)
        # calculate the combined inputs into final layer
        final_inputs = np.dot(a=self.weights_ho, b=hidden_outputs)
        # calculate the outputs from the final layer
        final_outputs = self.activation_function(x=final_inputs)

        return final_outputs
    

# create an instance of a neuralNetwork
nn1 = neuralNetwork(input_nodes=3, hidden_nodes=3, output_nodes=3, learning_rate=0.3)
model_output = nn1.query(list_input=[1.0, 0.5, -1.5])
print(model_output)



