import tensorflow as tf
from tensorflow.keras.applications import efficientnet


class Eff_UNet(tf.Model):

    # def __init__(self,
    #       extra_num = 1,
    #       dec_ch = [32, 64, 128, 256, 1024],
    #       stride = 32,
    #       net='b5',
    #       bot1x1=False,
    #       glob=False,
    #       bn = False,
    #       aspp=False,
    #       ocr=False,
    #       aux = False
    # ):

    def __init__(self, net):
        super().__init__()

        self.net = net
        # self.extra_num = extra_num
        # self.stride = stride
        # self.bot1x1 = bot1x1
        # self.glob = glob
        # self.bn = bn
        # self.aspp = aspp
        # self.ocr = ocr
        # self.aux = aux

        if net == 'b4':
            channel_multiplier = 1.4
            depth_multiplier = 1.8
            base_mode = efficientnet.EfficientNetB4()

        elif net == 'b5':
            channel_multiplier = 1.4
            depth_multiplier = 1.8
            base_mode = efficientnet.EfficientNetB5()

        elif net == 'b6':
            channel_multiplier = 1.4
            depth_multiplier = 1.8
            base_mode = efficientnet.EfficientNetB6()

        elif net == 'b7':
            channel_multiplier = 1.4
            depth_multiplier = 1.8
            base_mode = efficientnet.EfficientNetB7()

    def separate_blocks(eff_arch):

        
        model_layers = {}
        model_layers['head'] = eff_arch.layers[:7]
        model_layers['output'] = eff_arch.layers[-6:]

        
        for layer in eff_arch[7:-6]:
            pass

        pass

    @staticmethod
    def create_model(eff_base='B0'):
        """

        """

        base_model = None
        pass
