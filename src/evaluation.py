import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src import constants
from COCO_utils import process_data_utils


def show_train_set_and_gt(model, path):
    data, masks = process_data_utils.read_h5(path)
    predicted_ = model.predict(data[0:100])

    for i, mask in enumerate(predicted_):
        mask = np.argmax(mask, axis=-1)
        plt.imshow(mask)
        plt.show()


def evaluate_single(model, path):
    valid_files = os.listdir(path)
    data_arr = np.zeros((len(valid_files), constants.HEIGHT, constants.WIDTH, constants.CHANNELS), dtype=np.float64)

    for i, f in enumerate(valid_files):
        print(f)
        f = os.path.join(path, f)

        img = cv2.imread(f)
        data_arr[i] = cv2.resize(img, (constants.HEIGHT, constants.WIDTH))

        predicted_results = model.predict(data_arr)
        result_decode = np.argmax(predicted_results[i], axis=-1)
        print(result_decode)

        plt.figure(f)
        plt.imshow(result_decode)
        plt.show()


if __name__ == '__main__':
    model_path = '../model/basic_U_net_model/202007281101.h5'

    #
    valid_path = r'E:\extracted_data\COCO\images\val2017'

    train_path = '../data/train_car_truck_bus_128_128.h5'

    model = load_model(model_path)

    model.summary()

    # evaluate one by one
    # evaluate_single(model, valid_path)

    # test evaluate on training set
    show_train_set_and_gt(model, train_path)
