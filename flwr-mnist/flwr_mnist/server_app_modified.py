"""flwr-mnist: A Flower / TensorFlow app."""
import tensorflow as tf
from tensorflow import keras
from task import load_data 
from constant import num_clients , num_rounds , num_malicious_clients , num_random_indices
from client_app import client_fn
import random
import numpy as np
import matplotlib.pyplot as plt

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
    # Read from config
    
    model = keras.models.load_model("MNIST_Controller_Model.keras")
    
    # Constant values from constants.py
    n_rounds = num_rounds
    n_clients = num_clients
    num_test_images = num_random_indices
    malicious_clients = num_malicious_clients

    # Load data
    x_train, y_train, x_test, y_test = load_data()
    len_train = len(x_train) // n_clients
    random_indices = random.sample(range(len(x_test)), num_test_images)
    x_test = x_test[random_indices]
    x_test = np.array(x_test)
    y_test = y_test[random_indices]
    y_test = np.array(y_test)

    # Initialize list to store accuracies for plotting
    accuracies = []
    rounds = []

    for i in range(1, n_rounds + 1): 
        print("Starting round", i)
        client_weights = []
        parameters = model.get_weights()
        for i in range(n_clients):
 
            subset_images = x_train[i * len_train:(i + 1) * len_train]
            subset_labels = y_train[i * len_train:(i + 1) * len_train]
            
            if i < malicious_clients:
                subset_labels = [label_flip[y] for y in subset_labels]

            subset_images = np.array(subset_images)
            subset_labels = np.array(subset_labels)   
            data = subset_images, subset_labels, x_test, y_test
            
            weights = client_fn(parameters , data)

            client_weights.append(weights)

        num_layers = len(client_weights[0]) # Computes how many layers are there in the model
        global_weights = [0] * num_layers # Initializes the global weights to 0
        for layer in range(num_layers):
            weighted_sum = sum(
                client_weights[i][layer] for i in range(len(client_weights))
            ) / len(client_weights)
            global_weights[layer] = weighted_sum
        
        model.set_weights(global_weights)
        loss, accuracy = model.evaluate(x_test, y_test)
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)

        # Append accuracy every 10 rounds
        if i % 10 == 0:
            accuracies.append(accuracy)
            rounds.append(i)

    # Plot accuracy vs. rounds
    plt.plot(rounds, accuracies, marker='o')
    plt.title("Model Accuracy vs. Number of Rounds")
    plt.xlabel("Number of Rounds")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    server_fn()
