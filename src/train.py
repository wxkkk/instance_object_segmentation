import time
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from src import constants, basic_U_net_model
from COCO_utils import process_data_utils

train_data_path = '../data/train_2000.h5'

if __name__ == '__main__':
    train_images, train_masks = process_data_utils.read_h5(train_data_path)
    train_masks = to_categorical(train_masks, len(constants.CATE_LIST) + 1)
    print('train_shape:', train_images.shape)

    model = basic_U_net_model.build_model()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.categorical_accuracy, tf.keras.metrics.Recall()])

    cur_time = int(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
    model_path = '../model/basic_U_net_model/{}.h5'.format(cur_time)
    log_path = r'..\log\basic_U_net_log\{}'.format(cur_time)

    model_saver = ModelCheckpoint(
        filepath=model_path
        # save_best_only=True
    )

    early_stopper = EarlyStopping(
        monitor='loss',
        patience=100,
        mode='min',
        restore_best_weights=True
    )

    tensor_board = TensorBoard(
        log_dir=log_path
    )

    result = model.fit(
        train_images,
        train_masks,
        validation_split=0.1,
        verbose=2,
        epochs=100,
        batch_size=64,
        callbacks=[model_saver, early_stopper, tensor_board]
    )
