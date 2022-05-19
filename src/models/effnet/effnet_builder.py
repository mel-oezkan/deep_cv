"""
Model Builder used to create the base Efficient Net 
model for the U-Net implementation based on this paper:

https://openaccess.thecvf.com/content_CVPRW_2020/papers/w22/Baheti_Eff-UNet_A_Novel_Architecture_for_Semantic_Segmentation_in_Unstructured_Environment_CVPRW_2020_paper.pdf
"""

import tensorflow as tf


def efficientnet_params(model_name):
    """Get efficientnet params based on model name."""

    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }

    return params_dict[model_name]


class EffDecoder(tf.keras.Model):

    def separate_blocks(eff_arch: tf.keras.Model) -> dict:
        """Takes the architecture of the specific eff net
        and separates them into the 7 different blocks with 
        their respective in- and output layers. 

        :param eff_arch (tf.Model): Efficient net Model
        :returns (dict): Model comprised as a dictionary 
        """

        model_layers = {}
        model_layers['head'] = eff_arch.layers[:7]
        model_layers['output'] = eff_arch.layers[-6:]

        blocks = {f'block_{i}': [] for i in range(1, 8)}
        for layer in eff_arch[7:-6]:
            block_name = layer.name[:7]
            block_id = block_name[-2]

            blocks[f'bloock_{block_id}'].append(layer)

        model_layers.update(blocks)
        return model_layers

    def call(
        self,
        inputs,
        training=True,
    ):

        # Todo:
        # Iterate over the model_layer dict and save the ouputs of the
        # respective layers in a list such that the decoder is able to
        # access the outputs of the previous layers (skip connections)

        # Todo: Check if this will create a tf.graph

        x = tf.keras.layers.Input(inputs)

        return x
