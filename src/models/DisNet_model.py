from keras.layers import Dense, Activation, InputLayer, Input
from keras.models import Sequential
from keras.optimizers import Adam

def construct_DisNet_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(6,)))
    model.add(Dense(100, activation="selu"))
    model.add(Dense(100, activation="selu"))
    model.add(Dense(100, activation="selu"))
    model.add(Dense(1, activation="selu"))
    optimizer = Adam(1e-4)
    model.compile(optimizer=optimizer,
                  loss="mean_absolute_error")
    return model

