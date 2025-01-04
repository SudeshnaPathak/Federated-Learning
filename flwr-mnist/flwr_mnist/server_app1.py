"""flwr-mnist: A Flower / TensorFlow app."""
import tensorflow as tf
from tensorflow import keras
from task import load_data 
from constant import num_clients , num_rounds , num_malicious_clients , num_random_indices
from client_app import client_fn
import random
import numpy as np


label_flip = {
  0: 6,
  1: 1,
  2: 2,
  3: 3,
  4: 4,
  5: 5,
  6: 0,
  7: 7,
  8: 8,
  9: 9
}

def server_fn():
    model = keras.models.load_model("MNIST_Controller_Model.keras")
    
    x_train, y_train, x_test, y_test = load_data()
    len_train = len(x_train) // num_clients
    random_indices = random.sample(range(len(x_test)), num_random_indices)
    x_test, y_test = x_test[random_indices], y_test[random_indices]

    for i in range(1 , num_rounds + 1):
        print(f"Starting round {i}")
        client_weights = []
        parameters = model.get_weights()

        for i in range(num_clients):
            subset_images = x_train[i * len_train:(i + 1) * len_train]
            subset_labels = y_train[i * len_train:(i + 1) * len_train]

            if i < num_malicious_clients:
                subset_labels = [label_flip[y] for y in subset_labels]

            subset_images, subset_labels = np.array(subset_images), np.array(subset_labels)
            data = (subset_images, subset_labels, x_test, y_test)

            weights = client_fn(parameters, data)
            client_weights.append(weights)

            # Free memory
            del subset_images, subset_labels, data, weights
            tf.keras.backend.clear_session()

        # Aggregate weights
        num_layers = len(client_weights[0])
        global_weights = [np.mean([client_weights[k][layer] for k in range(len(client_weights))], axis=0) for layer in range(num_layers)]
        model.set_weights(global_weights)

        model.set_weights(global_weights)

        # Evaluate model
        loss, accuracy = model.evaluate(x_test, y_test)
        print(f"Round {i}: Loss = {loss}, Accuracy = {accuracy}")

if __name__ == "__main__":
    server_fn()
