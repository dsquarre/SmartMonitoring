import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class Model:
    def __init__(self, path, batch_size=32, model=None, initial_bias=None):
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

        self._train_step = None
        self._train_step_lr = None

        if model is not None:
            self.model = model
        else:
            self.model = self.build_model(initial_bias=initial_bias)

    def get_class_counts(self):
        """Returns local train partition total and positive sample counts for Round 0 handshake."""
        train_labels = np.array(self.y_train, dtype=np.int32)
        pos_count = int(np.sum(train_labels == 1))
        neg_count = int(np.sum(train_labels == 0))
        return {
            "total": len(train_labels),
            "pos": pos_count,
            "neg": neg_count
        }

    def build_model(self, initial_bias=None):
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
        
        # Classification Output Layer with dynamic or default prior bias
        bias_val = initial_bias if initial_bias is not None else -2.2
        bias_init = tf.keras.initializers.Constant(bias_val)
        output = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            bias_initializer=bias_init,
            name='afib'
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.0),
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
                # Per-record, per-lead z-score (chosen by the ECG calibration, simple/experiments/ecg_protocol.md)
                x_sample = ((x_sample - x_sample.mean(axis=0)) / np.maximum(x_sample.std(axis=0), 1e-6)).astype(np.float32)
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

    def _get_train_step(self, lr):
        """
        One compiled SGD step returning the per-sample losses. Plain SGD has no optimizer state,
        so every round starts from a fresh optimizer, as in the TMLR supplement's local_train.
        """
        if self._train_step is None or self._train_step_lr != lr:
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.0)
            model = self.model

            @tf.function(input_signature=[tf.TensorSpec([None, 1000, 12], tf.float32), tf.TensorSpec([None], tf.float32)])
            def step(X_batch, y_batch):
                with tf.GradientTape() as tape:
                    preds = model(X_batch, training=True)
                    per_sample = tf.keras.losses.binary_crossentropy(tf.expand_dims(y_batch, -1), preds)
                    loss = tf.reduce_mean(per_sample)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return per_sample

            self._train_step, self._train_step_lr = step, lr
        return self._train_step

    def train(self, epochs=5, lr=0.1, batch_size=None):
        """
        Local training as in the TMLR supplement (hrl_fl/client.py local_train): `epochs` epochs of plain SGD
        (no momentum) on unweighted binary cross-entropy. Returns the mean training loss and Oort's statistical
        utility |B| * sqrt(mean per-sample loss^2) over every sample seen during training.
        """
        if batch_size:
            self.batch_size = int(batch_size)
        step = self._get_train_step(float(lr))
        loss_sum, sq_sum, seen, batches = 0.0, 0.0, 0, 0
        for _ in range(int(epochs)):
            for X_batch, y_batch in self._data_generator(self.X_train, self.y_train, shuffle=True):
                per_sample = step(tf.constant(X_batch), tf.constant(y_batch)).numpy().astype(np.float64)
                loss_sum += float(per_sample.mean())
                sq_sum += float(np.sum(per_sample ** 2))
                seen += len(per_sample)
                batches += 1
        if batches == 0:
            return {"train_loss": 0.0, "stat_utility": 0.0}
        return {"train_loss": loss_sum / batches,
                "stat_utility": float(self.num_train_samples * np.sqrt(sq_sum / seen))}

    def evaluate(self, threshold=0.3, return_preds=False):
        """Evaluates model performance on the local test partition and returns metrics."""
        empty_res = {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "roc_auc": 0.5,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0
        }
        if self.num_test_samples == 0:
            if return_preds:
                return empty_res, np.array([], dtype=int), np.array([], dtype=np.float32)
            return empty_res

        test_ds = self.get_dataset(self.X_test, self.y_test, shuffle=False)
        steps = max(1, int(np.ceil(self.num_test_samples / self.batch_size)))

        preds = self.model.predict(test_ds, steps=steps, verbose=0)
        preds_flat = preds.flatten()[:self.num_test_samples]
        pred_binary = (preds_flat >= threshold).astype(int)

        y_true = np.array(self.y_test[:self.num_test_samples], dtype=int)

        eps = 1e-7
        preds_clipped = np.clip(preds_flat, eps, 1.0 - eps)
        loss_val = float(-np.mean(y_true * np.log(preds_clipped) + (1.0 - y_true) * np.log(1.0 - preds_clipped)))

        unique_labels = np.unique(y_true)
        if len(unique_labels) > 1:
            auc_val = float(roc_auc_score(y_true, preds_flat))
        else:
            auc_val = 0.5

        # Hard threshold 0.3 metrics (Primary clinical/paper reporting standard)
        prec = float(precision_score(y_true, pred_binary, zero_division=0))
        rec = float(recall_score(y_true, pred_binary, zero_division=0))
        denom_f2 = (4 * prec + rec)
        f2_val = float((5 * prec * rec) / denom_f2) if denom_f2 > 0 else 0.0

        tp_val = int(np.sum((y_true == 1) & (pred_binary == 1)))
        tn_val = int(np.sum((y_true == 0) & (pred_binary == 0)))
        fp_val = int(np.sum((y_true == 0) & (pred_binary == 1)))
        fn_val = int(np.sum((y_true == 1) & (pred_binary == 0)))

        metrics_dict = {
            "loss": loss_val,
            "accuracy": float(accuracy_score(y_true, pred_binary)),
            "precision": prec,
            "recall": rec,
            "f1": float(f1_score(y_true, pred_binary, zero_division=0)),
            "f2": f2_val,
            "roc_auc": auc_val,
            "tp": tp_val,
            "fp": fp_val,
            "tn": tn_val,
            "fn": fn_val
        }

        if return_preds:
            return metrics_dict, y_true, preds_flat
        return metrics_dict

    def close(self):
        """Explicit memory cleanup for dataset files and model references."""
        if hasattr(self, 'data') and self.data is not None:
            try:
                self.data.close()
            except Exception:
                pass
            self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_samples(self):
        return self.num_train_samples