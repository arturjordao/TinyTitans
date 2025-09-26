import tensorflow as tf
import numpy as np

class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, init_lr=0.01, schedule=[(25, 1e-2), (50, 1e-3), (100, 1e-4)]):
        super().__init__()
        self.init_lr = init_lr
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        lr = self.init_lr
        for i in range(len(self.schedule) - 1):
            if epoch >= self.schedule[i][0] and epoch < self.schedule[i + 1][0]:
                lr = self.schedule[i][1]

        if epoch >= self.schedule[-1][0]:
            lr = self.schedule[-1][1]

        print(f'Learning rate: {lr}')
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
