import tensorflow as tf
import numpy as np

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    f1_score
)


class Model:

    def __init__(self,path):
        data = np.load(path)

        X = data['X']

        y_anom = data['y_anomaly']

        y_dis = data['y_disease']

        (

            self.X_train,
            self.X_test,

            self.y_anom_train,
            self.y_anom_test,

            self.y_dis_train,
            self.y_dis_test

        ) = train_test_split(

            X,
            y_anom,
            y_dis,

            test_size=0.21,

            random_state=93
        )

        if self.X_train.shape[1] == 6:

            self.X_train = np.transpose(
                self.X_train,
                (0,2,1)
            )

            self.X_test = np.transpose(
                self.X_test,
                (0,2,1)
            )

        self.X_train = np.nan_to_num(
            self.X_train
        ).astype(np.float32)

        self.X_test = np.nan_to_num(
            self.X_test
        ).astype(np.float32)
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

        x = tf.keras.layers.Bidirectional(

            tf.keras.layers.LSTM(
                128,
                return_sequences=True
            )

        )(x)

        attn = tf.keras.layers.Dense(
            1,
            activation='tanh'
        )(x)

        attn = tf.keras.layers.Softmax(
            axis=1
        )(attn)

        x = x * attn

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

    def train(self, epochs=1):

        self.model.fit(

            self.X_train,

            {
                'anomaly': self.y_anom_train,
                'disease': self.y_dis_train
            },

            epochs=epochs,

            batch_size=32,

            verbose=1
        )

    def evaluate(self):

        pred_anom, pred_dis = self.model.predict(
            self.X_test,
            verbose=0
        )

        pred_anom = (
            pred_anom > 0.5
        ).astype(int).flatten()

        pred_dis = np.argmax(
            pred_dis,
            axis=1
        )

        true_dis = np.argmax(
            self.y_dis_test,
            axis=1
        )

        metrics = {

            "anomaly_accuracy":

            accuracy_score(
                self.y_anom_test,
                pred_anom
            ),

            "disease_accuracy":

            accuracy_score(
                true_dis,
                pred_dis
            ),

            "disease_f1":

            f1_score(
                true_dis,
                pred_dis,
                average='weighted'
            )
        }

        return metrics

    def get_samples(self):

        return len(self.X_train)
