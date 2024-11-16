# Description: Custom callback to stop training when validation loss is below a threshold
# Author: Balaji Anna
# Last Edited: 2024-Nov-16
import tensorflow as tf

class ThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(ThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.threshold:
            print(f"\nValidation loss {val_loss} is below threshold {self.threshold}. Stopping training.")
            self.model.stop_training = True