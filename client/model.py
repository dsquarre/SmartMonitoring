import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score
)


class Model:

    def __init__(self, path):

        self.data = np.load(
            path,
            mmap_mode='r'
        )

        self.X = self.data['X']
        self.y_anom = self.data['y_anomaly']
        self.y_dis = self.data['y_disease']

        self.num_samples = len(self.X)

        split_index = int(0.8 * self.num_samples)

        self.train_indices = np.arange(0, split_index)
        self.test_indices = np.arange(split_index, self.num_samples)

        self.model = self.build_model()

    def build_model(self):

        inputs = tf.keras.Input(shape=(1000, 6))

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

    def train_generator(self, batch_size=16):

        indices = np.array(self.train_indices)

        np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):

            batch_idx = indices[start:start + batch_size]

            X_batch = []
            y_anom_batch = []
            y_dis_batch = []

            for idx in batch_idx:

                x = self.X[idx]

                if x.shape[0] == 6:
                    x = np.transpose(x, (1, 0))

                x = np.nan_to_num(x).astype(np.float32)

                X_batch.append(x)

                y_anom_batch.append(
                    np.float32(self.y_anom[idx])
                )

                y_dis_batch.append(
                self.y_dis[idx].astype(np.float32)
                )

            X_batch = np.array(X_batch)

            y_anom_batch = np.array(y_anom_batch)

            y_dis_batch = np.array(y_dis_batch)

            yield (
                X_batch,
                {
                "anomaly": y_anom_batch,
                "disease": y_dis_batch
                }
            )

    def test_generator(self):

        for idx in self.test_indices:

            x = self.X[idx]

            if x.shape[0] == 6:
                x = np.transpose(x, (1, 0))

            x = np.nan_to_num(x).astype(np.float32)

            y1 = np.float32(self.y_anom[idx])

            y2 = self.y_dis[idx].astype(np.float32)

            yield (
                x,
                {
                    "anomaly": y1,
                    "disease": y2
                }
            )

    def train(self, epochs=1):

        train_ds = tf.data.Dataset.from_generator(

            lambda: self.train_generator(batch_size=16),

            output_signature=(

                tf.TensorSpec(
                shape=(None, 1000, 6),
                dtype=tf.float32
                ),

                {
                "anomaly": tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.float32
                    ),

                "disease": tf.TensorSpec(
                    shape=(None, 4),
                    dtype=tf.float32
                    )
                }
            )
        )

        train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
        )

        for x_batch, y_batch in train_ds.take(1):

            print("\nBATCH CHECK")
            print("X batch shape:", x_batch.shape)
            print("Anomaly batch shape:", y_batch["anomaly"].shape)
            print("Disease batch shape:", y_batch["disease"].shape)

        steps_per_epoch = len(self.train_indices) // 16

        self.model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1
        )

    def evaluate(self):

        test_ds = tf.data.Dataset.from_generator(

            self.test_generator,

            output_signature=(

                tf.TensorSpec(
                    shape=(1000, 6),
                    dtype=tf.float32
                ),

                {
                    "anomaly": tf.TensorSpec(
                        shape=(),
                        dtype=tf.float32
                    ),

                    "disease": tf.TensorSpec(
                        shape=(4,),
                        dtype=tf.float32
                    )
                }
            )
        )

        test_ds = test_ds.batch(16)

        pred_anom, pred_dis = self.model.predict(
            test_ds,
            verbose=0
        )

        pred_anom = (
            pred_anom > 0.5
        ).astype(int).flatten()

        pred_dis = np.argmax(
            pred_dis,
            axis=1
        )

        true_anom = self.y_anom[self.test_indices]

        true_dis = np.argmax(
            self.y_dis[self.test_indices],
            axis=1
        )

        metrics = {

            "anomaly_accuracy": accuracy_score(
                true_anom,
                pred_anom
            ),

            "disease_accuracy": accuracy_score(
                true_dis,
                pred_dis
            ),

            "disease_f1": f1_score(
                true_dis,
                pred_dis,
                average='weighted'
            )
        }

        return metrics

    def get_weights(self):

        return self.model.get_weights()

    def set_weights(self, weights):

        self.model.set_weights(weights)

    def get_samples(self):

        return len(self.train_indices)