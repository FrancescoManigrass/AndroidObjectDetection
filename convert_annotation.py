from absl import app, flags, logging
import os
import pickle
from os import listdir
from os.path import isfile, join
from absl.flags import FLAGS
import cv2

OldTrainPath="data/dataset/train.txt"


def main(_argv):
    from core.yolov4 import YOLOv4, decode
    import tensorflow as tf
    import core.utils as utils
    import pickle

    input_layer = tf.keras.layers.Input([608, 608, 3])
    feature_maps = YOLOv4(input_layer, NUM_CLASS=13)
    model = tf.keras.Model(input_layer, feature_maps)
    utils.load_weights(model, "./yolov4-obj_26000.weights")
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    pickle.dump(weights, open("./yolov4.pkl", 'wb'))



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass