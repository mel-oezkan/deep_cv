import tensorflow as tf

# def dice_coefficient(y_true, y_pred, smooth=1):
#     intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
#     union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
#     dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
#     return dice

class MeanIOULoss(tf.keras.losses.Loss):

    def __init__(self, name="MeanIOU_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.dtypes.float64)
        y_pred = tf.cast(y_pred, tf.dtypes.float64)
        I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
        U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
        return tf.reduce_mean(I / U)

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1, name="dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth


    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.dtypes.float64)
        y_pred = tf.cast(y_pred, tf.dtypes.float64)

        intersection = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])  
        dice_loss = tf.reduce_mean((2. * intersection + self.smooth) / (union + self.smooth), axis=0)

        return dice_loss
        


class HybridLoss(tf.keras.losses.Loss):

    def __init__(self, smooth=1, name="hybrid_loss"):
        super().__init__(name=name)

        self.smooth = smooth
        self.dice_fnc = DiceLoss(smooth)
        self.meanIOU_fnc = MeanIOULoss()


    def call(self, y_true, y_pred):
        
        loss_dice = self.dice_fnc(y_true, y_pred)
        loss_iou = self.meanIOU_fnc(y_true, y_pred)  

        return loss_dice + loss_iou 
