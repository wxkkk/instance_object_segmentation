from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate, MaxPool2D
from src import constants


def build_model():

    inputs = Input(shape=constants.INPUT_SHAPE)

    C1 = Conv2D(filters=2, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    P1 = MaxPool2D((2, 2))(C1)

    C2 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(P1)
    P2 = MaxPool2D((2, 2))(C2)

    C3 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(P2)

    # Deconv
    U4 = Conv2DTranspose(filters=4, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(C3)
    U4 = concatenate([U4, C2])
    C4 = Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='same')(U4)

    U5 = Conv2DTranspose(filters=2, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same')(C4)
    U5 = concatenate([U5, C1])
    C5 = Conv2D(filters=2, kernel_size=(3, 3), activation='relu', padding='same')(U5)

    outputs = Conv2D(len(constants.CATE_LIST) + 1, (1, 1), activation='softmax')(C5)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.summary()

    return model


if __name__ == '__main__':
    build_model()

