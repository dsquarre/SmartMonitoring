import tensorflow as tf
import numpy as np

class Model:

    def __init__(self):
        self.model = self.build_model()
    def build_model(self):
        inputs = tf.keras.Input(shape=(1000, 12), name='ecg_input')

        # Block 1
        x = tf.keras.layers.Conv1D(64, kernel_size=15, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)

        # Block 2
        x = tf.keras.layers.Conv1D(128, kernel_size=7, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)

        # Block 3
        x = tf.keras.layers.Conv1D(256, kernel_size=5, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Classification Head
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid', name='afib')(x)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn')
            ]
        )
        return model