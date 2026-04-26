import os
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
from models.DisNet_model import construct_DisNet_model

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from config import classes_config

class DistanceByDisNet:
    def __init__(self, model_path = "model/best_disnet_model.keras"):
        if os.path.exists(model_path):
            print("Continue training from checkpoints ...")
            self.model = load_model(model_path)
        else:
            print("No model checkpoints founded, construct new model...")
            self.model = construct_DisNet_model()

        self.callbacks = [EarlyStopping(monitor='val_loss', patience=200, verbose=1), 
            ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)]

    def get_X_from_box(self, box, img_shape):
        width = (box.xywh.tolist()[0][2] / img_shape[1])
        height = (box.xywh.tolist()[0][3] / img_shape[0])
        diagonal = np.sqrt(np.square(width) + np.square(height))
        class_h = classes_config[int(box.cls.item())]["class_h"]
        class_w = classes_config[int(box.cls.item())]["class_w"]
        class_d = classes_config[int(box.cls.item())]["class_d"]

        return np.array([[1 / width, 1 / height, 1 / diagonal, class_h, class_w, class_d]])

    def predict(self, box, img_shape):
        return self.model.predict(self.get_X_from_box(box, img_shape))[0][0]

    def learn(self, X_train, y_train, X_val, y_val, epochs=10, verbose=1, batch_size=50):
        return self.model.fit(x=X_train, y=y_train, epochs=epochs,
                        callbacks=self.callbacks, verbose=verbose, batch_size=batch_size,
                        validation_data=(X_val, y_val))

def distance_by_DisNet(DisNet_model, boxes, img_shape):
    distances = []

    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xywh}")

        d = DisNet_model.predict(box, img_shape)
        # print(d)
        distances.append(d)

    return distances
