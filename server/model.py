import tensorflow as tf
import numpy as np

class Model:

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):

        inputs = tf.keras.Input(
            shape=(1000, 6)
        )

        x = tf.keras.layers.Conv1D(
            64,
            7,
            padding='same',
            activation='relu'
        )(inputs)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling1D(2)(x)

        x = tf.keras.layers.Conv1D(
            128,
            5,
            padding='same',
            activation='relu'
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling1D(2)(x)

        x = tf.keras.layers.Conv1D(
            256,
            3,
            padding='same',
            activation='relu'
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling1D(2)(x)

        x = tf.keras.layers.SeparableConv1D(256,3,padding='same',activation='relu')(x)

        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = tf.keras.layers.Dense(
            128,
            activation='relu'
        )(x)

        x = tf.keras.layers.Dropout(0.4)(x)

        anomaly_output = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            name='anomaly'
        )(x)

        disease_output = tf.keras.layers.Dense(
            4,
            activation='softmax',
            name='disease'
        )(x)

        model = tf.keras.Model(
            inputs,
            [anomaly_output, disease_output]
        )

        model.compile(

            optimizer=tf.keras.optimizers.Adam(
                learning_rate=1e-4,
                clipnorm=1.0
            ),

            loss={
                'anomaly': 'binary_crossentropy',
                'disease': 'categorical_crossentropy'
            },

            metrics={
                'anomaly': 'accuracy',
                'disease': 'categorical_accuracy'
            }
        )

        return model