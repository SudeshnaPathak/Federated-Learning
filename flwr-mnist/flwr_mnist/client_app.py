"""flwr-mnist: A Flower / TensorFlow app."""

from task import load_data, load_model
from constant import local_epochs , batchsize , verbose_val
import tensorflow as tf
from tensorflow import keras

client_counter = 0
# Define Flower Client and client_fn
class Client():
    def __init__(
        self, model, data, epochs, batch_size, verbose , parameters
    ):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.parameters = parameters

    def fit(self):
        self.model.set_weights(self.parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self):
        self.model.set_weights(self.parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(parameters , data):
    # Load model and data
    global client_counter
    net = load_model()
    epochs = local_epochs
    batch_size = batchsize
    verbose = verbose_val
    # Return Client instance
    client_counter += 1
    client =  Client(
        net, data, epochs, batch_size, verbose , parameters
    )
    print(f"Client : {client_counter}")
    weights, _, _ = client.fit()
    return weights
    


# if __name__ == "__main__":
#     client = client_fn()
#     weights, train_samples, train_metrics = client.fit()
#     loss, test_samples, test_metrics = client.evaluate()
#     print(f"weights: {weights}")
#     print(f"train_samples: {train_samples}")
#     print(f"train_metrics: {train_metrics}")
#     print(f"loss: {loss}")
#     print(f"test_samples: {test_samples}")
#     print(f"test_metrics: {test_metrics}")
    
