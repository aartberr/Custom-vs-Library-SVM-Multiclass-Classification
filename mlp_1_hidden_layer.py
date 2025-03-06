import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras

np.random.seed(42)

def one_hot_encode_labels(df, num_classes):
  # Use one-hot encoding
  labels = np.zeros((df.shape[0], num_classes))
  # Set the label for each row
  for i, label in enumerate(df['label']):
    labels[i, int(label)] = 1

  return labels

# Format and batch the data
def df_to_dataset(dataframe, labels, batch_size = 1024):
    df = dataframe.copy()
    df = df['text']
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def mlp_1hidden_kayer_NN(train, train_labels, test, test_labels, embedding, learning_rate, epochs, batch_size, layer_size):
    train_data = df_to_dataset(train, train_labels, batch_size)
    test_data = df_to_dataset(test, test_labels, batch_size)

    # Model
    model = tf_keras.Sequential()

    # Embedding Layer
    hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

    # Layers
    model.add(hub_layer) #text -> vector
    model.add(tf_keras.layers.Dense(layer_size , activation = 'relu')) # 1st hidden layer

    # Output Layer
    model.add(tf_keras.layers.Dense(num_classes, activation = 'softmax'))

    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate = learning_rate), loss=tf_keras.losses.Hinge(reduction='sum_over_batch_size', name='hinge'),metrics = ['accuracy'])

    history = model.fit(train_data, epochs = epochs, validation_data = test_data)

    # Collect results
    results = {
        "epoch": list(range(1, epochs + 1)),
        "train_loss": history.history['loss'],
        "train_accuracy": history.history['accuracy'],
        "test_loss": history.history['val_loss'],
        "test_accuracy": history.history['val_accuracy'],
    }

    results_df = pd.DataFrame(results)

    return results_df

# Read the dataset file
dataframe = pd.read_csv('dataset1.csv')
# sad(0), joy(1), love(2), anger(3), fear(4), surprise(5)
num_classes = 6
df = dataframe.dropna()
df = df.head(100000)

# Find the number of samples (N) in the smallest class
class_counts = df['label'].value_counts()
N = class_counts.min()

# From each class keep only N samples
df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(N, random_state=42))

# Split the data
train, test = np.split(df.sample(frac=1), [int(0.6 * len(df))])
train_labels = one_hot_encode_labels(train, num_classes)
test_labels = one_hot_encode_labels(test, num_classes)

#Default values
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
learning_rate = 0.001
epochs = 20
batch_size = 512
layer_size = 32

# Run the MLP with 1 hidden layer model
lib_result = mlp_1hidden_kayer_NN(train, train_labels, test, test_labels, embedding, learning_rate = learning_rate, epochs = epochs, batch_size = batch_size, layer_size = layer_size)
