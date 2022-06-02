import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D

from src.models.basicUnet.basicBuilder import DecoderMiniBlock, EncoderMiniBlock


class BasicUnet():

    @staticmethod
    def create_model(input_size=(128, 128, 3), n_filters=32, n_classes=2):
        """
            Combine both encoder and decoder blocks according to the U-Net research paper
            Return the model as output 
            """
        # Input size represent the size of 1 image (the size used for pre-processing)
        inputs = Input(input_size)

        # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
        # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image
        cblock1 = EncoderMiniBlock(
            inputs, n_filters, dropout_prob=0, max_pooling=True)
        cblock2 = EncoderMiniBlock(
            cblock1[0], n_filters*2, dropout_prob=0, max_pooling=True)
        cblock3 = EncoderMiniBlock(
            cblock2[0], n_filters*4, dropout_prob=0, max_pooling=True)
        cblock4 = EncoderMiniBlock(
            cblock3[0], n_filters*8, dropout_prob=0.3, max_pooling=True)
        cblock5 = EncoderMiniBlock(
            cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

        # Decoder includes multiple mini blocks with decreasing number of filters
        # Observe the skip connections from the encoder are given as input to the decoder
        # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
        ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8)
        ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
        ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
        ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)

        # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
        # Followed by a 1x1 Conv layer to get the image to the desired size.
        # Observe the number of channels will be equal to number of output classes
        conv9 = Conv2D(n_filters,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(ublock9)

        conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

        # Define the model
        model = tf.keras.Model(inputs=inputs, outputs=conv10)

        return model
