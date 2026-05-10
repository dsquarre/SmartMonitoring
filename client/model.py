import tensorflow as tf
import numpy as np

class Model:

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):

        inputs = tf.keras.Input(shape=(1000, 6))

        x = tf.keras.layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

        x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

        x = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True)
        )(x)

        attn = tf.keras.layers.Dense(1, activation='tanh')(x)
        attn = tf.keras.layers.Softmax(axis=1)(attn)

        x = x * attn
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        anomaly_output = tf.keras.layers.Dense(
            1, activation='sigmoid', name='anomaly'
        )(x)

        disease_output = tf.keras.layers.Dense(
            4, activation='softmax', name='disease'
        )(x)

        model = tf.keras.Model(inputs, [anomaly_output, disease_output])

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

    def fit(self, X_train, y_anom_train, y_dis_train):

        if X_train.shape[1] == 6:
            X_train = np.transpose(X_train, (0, 2, 1))

        X_train = np.nan_to_num(X_train, nan=0.0)

        history = self.model.fit(
            X_train,
            {
                'anomaly': y_anom_train,
                'disease': y_dis_train
            },
            epochs=10,
            batch_size=32,
            validation_split=0.1
        )

        return history
    
    def eval(self, X_test, y_anom_test, y_dis_test):

        if X_test.shape[1] == 6:
            X_test = np.transpose(X_test, (0, 2, 1))

        X_test = np.nan_to_num(X_test, nan=0.0)

        results = self.model.evaluate(
            X_test,
            {
                'anomaly': y_anom_test,
                'disease': y_dis_test
            }
        )

        pred_anom, pred_dis = self.model.predict(X_test)

        pred_anom = (pred_anom > 0.5).astype(int).flatten()
        pred_dis = np.argmax(pred_dis, axis=1)

        true_anom = y_anom_test
        true_dis = np.argmax(y_dis_test, axis=1)

        from sklearn.metrics import classification_report

        print("\nAnomaly Report:")
        print(classification_report(true_anom, pred_anom))

        print("\nDisease Report:")
        print(classification_report(
            true_dis,
            pred_dis,
            target_names=['Normal','Arrhythmia','Apnea','AF']
        ))

        return results

    def train(self,epochs):
        #do model.fit() for given epochs
        self.model.save("client_model.h5")

    def evaluate(self):
        #do model.eval and return a dict of metrics
        return {'hi': 'hello'}
    def get_samples(self):
        #return number of samples used for training
        return 1000