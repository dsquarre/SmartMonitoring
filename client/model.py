import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight


class Model:
    def __init__(self, path, batch_size=32):
        self.dataset_name = path
        self.batch_size = batch_size

        # Memory-mapped access to client partition
        self.data = np.load(path, mmap_mode='r')
        self.X_train = self.data['X_train']
        self.y_train = self.data['y_train']
        self.X_test = self.data['X_test']
        self.y_test = self.data['y_test']

        self.num_train_samples = len(self.X_train)
        self.num_test_samples = len(self.X_test)

        # Calculate local class balancing weights
        train_labels = np.array(self.y_train, dtype=np.int32)
        unique_classes = np.unique(train_labels)
        if len(unique_classes) > 1:
            weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
            self.class_weights = {int(c): float(w) for c, w in zip(unique_classes, weights)}
            self.pos_weight = float(weights[1])
            self.neg_weight = float(weights[0])
        else:
            self.class_weights = {0: 1.0, 1: 1.0}
            self.pos_weight = 1.0
            self.neg_weight = 1.0

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

    def _data_generator(self, X, y, shuffle=True):
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            X_batch = []
            y_batch = []

            for idx in batch_idx:
                x_sample = X[idx]
                if x_sample.shape[0] == 12 and x_sample.shape[1] != 12:
                    x_sample = np.transpose(x_sample, (1, 0))

                x_sample = np.nan_to_num(x_sample, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                y_sample = np.float32(y[idx])

                X_batch.append(x_sample)
                y_batch.append(y_sample)

            yield np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

    def get_dataset(self, X, y, shuffle=True):
        dataset = tf.data.Dataset.from_generator(
            lambda: self._data_generator(X, y, shuffle=shuffle),
            output_signature=(
                tf.TensorSpec(shape=(None, 1000, 12), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    def train(self, epochs=1, verbose=0):
        """Standard local training for FedAvg, FedProx, etc."""
        train_ds = self.get_dataset(self.X_train, self.y_train, shuffle=True)
        steps_per_epoch = max(1, int(np.ceil(self.num_train_samples / self.batch_size)))

        history = self.model.fit(
            train_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            class_weight=self.class_weights,
            verbose=verbose
        )
        return history

    def train_local_gradients_fv(self):
        """
        Computes accumulated raw gradients across the local dataset for FedFV aggregation.
        """
        accumulated_grads = [np.zeros(var.shape, dtype=np.float32) for var in self.model.trainable_variables]
        total_loss = 0.0
        batch_count = 0

        total_steps = max(1, int(np.ceil(self.num_train_samples / self.batch_size)))

        for X_batch, y_batch in self._data_generator(self.X_train, self.y_train, shuffle=True):
            with tf.GradientTape() as tape:
                preds = self.model(X_batch, training=True)
                preds = tf.squeeze(preds, axis=-1)

                # Class-weighted binary crossentropy loss
                bce = tf.keras.losses.binary_crossentropy(y_batch, preds)
                sample_weights = tf.where(tf.equal(y_batch, 1.0), self.pos_weight, self.neg_weight)
                batch_loss = tf.reduce_mean(bce * sample_weights)

            raw_grads = tape.gradient(batch_loss, self.model.trainable_variables)

            for i, grad in enumerate(raw_grads):
                if grad is not None:
                    accumulated_grads[i] += grad.numpy()

            total_loss += float(batch_loss)
            batch_count += 1

        if batch_count == 0:
            return [np.zeros(var.shape, dtype=np.float32) for var in self.model.trainable_variables], 0.0

        average_grads = [g / batch_count for g in accumulated_grads]
        average_loss = total_loss / batch_count

        return average_grads, average_loss

    def apply_global_gradients_fv(self, global_gradients, server_lr=0.001):
        """Applies global updates directly to trainable variables."""
        native_gradients = [np.array(gg, dtype=np.float32) for gg in global_gradients]
        for var, gg in zip(self.model.trainable_variables, native_gradients):
            current_value = var.numpy()
            updated_value = current_value - (server_lr * gg)
            var.assign(updated_value)

    def evaluate(self):
        """Evaluates model performance on the local test partition and returns metrics."""
        if self.num_test_samples == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "roc_auc": 0.0
            }

        test_ds = self.get_dataset(self.X_test, self.y_test, shuffle=False)
        steps = max(1, int(np.ceil(self.num_test_samples / self.batch_size)))

        preds = self.model.predict(test_ds, steps=steps, verbose=0)
        preds_flat = preds.flatten()[:self.num_test_samples]
        pred_binary = (preds_flat >= 0.5).astype(int)

        y_true = np.array(self.y_test[:self.num_test_samples], dtype=int)

        eval_res = self.model.evaluate(test_ds, steps=steps, verbose=0, return_dict=True)

        unique_labels = np.unique(y_true)
        if len(unique_labels) > 1:
            auc_val = float(roc_auc_score(y_true, preds_flat))
        else:
            auc_val = 0.5

        return {
            "loss": float(eval_res.get("loss", 0.0)),
            "accuracy": float(accuracy_score(y_true, pred_binary)),
            "precision": float(precision_score(y_true, pred_binary, zero_division=0)),
            "recall": float(recall_score(y_true, pred_binary, zero_division=0)),
            "f1": float(f1_score(y_true, pred_binary, zero_division=0)),
            "roc_auc": auc_val,
            "tp": int(eval_res.get("tp", 0)),
            "fp": int(eval_res.get("fp", 0)),
            "tn": int(eval_res.get("tn", 0)),
            "fn": int(eval_res.get("fn", 0))
        }

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_samples(self):
        return self.num_train_samples