import tensorflow as tf
import matplotlib.pyplot as plt


class DisplayTestCallback(tf.keras.callbacks.Callback):

    # Todo:
    #   Create a wrapper that checks wether data is
    #   Tf.Dataset or is contained in two np.arrays

    def __init__(self, dataset, val_x=None, val_y=None) -> None:
        super(DisplayTestCallback).__init__()

        # Todo: wrap in method from todo above
        # only take 4 samples from the test data
        # self.sample_count = min(val_x.shape[0], 4)
        # self.val_x = val_x[:self.sample_count]
        # self.val_y = val_y[:self.sample_count]

        self.data = dataset

    def on_epoch_end(self, epoch, logs=None):
        self.show_pred()

    # function handling visualization with tf.dataset
    def show_pred(self):
        _, ax = plt.subplots(1, 3)

        for in_im, target_im in self.data:
            preds = self.model.predict(in_im)

            ax[0].imshow(in_im[0])
            ax[0].axis('off')

            ax[1].imshow(preds[0])
            ax[1].axis('off')

            ax[2].imshow(target_im[0])
            ax[2].axis('off')

            plt.show()
            break

    # def show_predictions(self):
    #     _, ax = plt.subplots(3, self.sample_count)

    #     preds = self.model.predict(self.val_x)
    #     if len(preds.shape) >= 3 and preds.ndim >= 2:
    #         for index, sample in enumerate(zip(self.val_x, preds, self.val_y)):
    #             ax[0, index].imshow(sample[0])
    #             ax[0, index].axis('off')
    #             ax[1, index].imshow(sample[1])
    #             ax[1, index].axis('off')
    #             ax[2, index].imshow(sample[2])
    #             ax[2, index].axis('off')

    #             plt.show()
    #             break
