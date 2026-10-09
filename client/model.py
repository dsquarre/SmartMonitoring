import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight


class Model:
    def __init__(self, path, batch_size=32, model=None, initial_bias=None, class_weights=None, data=None):
        self.dataset_name = path
        self.batch_size = batch_size

        # Client partition: pre-loaded in-memory arrays if given (dict with X_train/y_train/X_test/y_test),
        # otherwise memory-mapped access to the .npz file
        self.data = data if data is not None else np.load(path, mmap_mode='r')
        self.X_train = self.data['X_train']
        self.y_train = self.data['y_train']
        self.X_test = self.data['X_test']
        self.y_test = self.data['y_test']

        self.num_train_samples = len(self.X_train)
        self.num_test_samples = len(self.X_test)

        # Set or calculate class balancing weights
        if class_weights is not None:
            self.class_weights = {int(k): float(v) for k, v in class_weights.items()}
            self.pos_weight = float(self.class_weights.get(1, 1.0))
            self.neg_weight = float(self.class_weights.get(0, 1.0))
        else:
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

            # Vectorized batch assembly (same values and order as per-sample processing)
            X_batch = np.asarray(X[batch_idx])
            if X_batch.shape[1] == 12 and X_batch.shape[2] != 12:
                X_batch = np.transpose(X_batch, (0, 2, 1))
            X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            y_batch = np.asarray(y[batch_idx]).astype(np.float32)

            yield X_batch, y_batch

    @staticmethod
    def _to_array(X):
        """Loads a client partition into memory as float32 (N, 1000, 12) with NaN/Inf zeroed."""
        X_arr = np.asarray(X)
        if X_arr.ndim == 3 and X_arr.shape[1] == 12 and X_arr.shape[2] != 12:
            X_arr = np.transpose(X_arr, (0, 2, 1))
        return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def get_dataset(self, X, y, shuffle=True):
        dataset = tf.data.Dataset.from_generator(
            lambda: self._data_generator(X, y, shuffle=shuffle),
            output_signature=(
                tf.TensorSpec(shape=(None, 1000, 12), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    def train(self, epochs=5, verbose=0):
        """Standard local training for FedAvg, FedProx, etc."""
        # Client partitions are small (~200 ECGs), so train from in-memory arrays (reshuffled every epoch)
        X_train = self._to_array(self.X_train)
        y_train = np.asarray(self.y_train).astype(np.float32)

        history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            shuffle=True,
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
            "fn": 0,
            # Training-reward soft confusion constructs (not clinical/paper reporting metrics)
            "soft_tp": 0.0,
            "soft_fp": 0.0,
            "soft_tn": 0.0,
            "soft_fn": 0.0,
            "soft_precision": 0.0,
            "soft_recall": 0.0,
            "soft_f2": 0.0
        }
        if self.num_test_samples == 0:
            if return_preds:
                return empty_res, np.array([], dtype=int), np.array([], dtype=np.float32)
            return empty_res

        # Inference-mode forward pass on in-memory batches (avoids per-call tf.data/predict overhead;
        # batch size does not affect inference-mode outputs)
        X_test = self._to_array(self.X_test)
        eval_batch = max(self.batch_size, 256)
        preds = [
            np.asarray(self.model.predict_on_batch(X_test[start:start + eval_batch]))
            for start in range(0, self.num_test_samples, eval_batch)
        ]
        preds_flat = np.concatenate(preds).flatten()[:self.num_test_samples]
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

        tp_val = int(np.sum((y_true == 1) & (pred_binary == 1)))
        tn_val = int(np.sum((y_true == 0) & (pred_binary == 0)))
        fp_val = int(np.sum((y_true == 0) & (pred_binary == 1)))
        fn_val = int(np.sum((y_true == 1) & (pred_binary == 0)))

        # Hard threshold 0.3 metrics (Primary clinical/paper reporting standard), computed from the
        # confusion counts (identical to sklearn precision/recall/f1 with zero_division=0)
        prec = float(tp_val / (tp_val + fp_val)) if (tp_val + fp_val) > 0 else 0.0
        rec = float(tp_val / (tp_val + fn_val)) if (tp_val + fn_val) > 0 else 0.0
        f1_val = float(2 * tp_val / (2 * tp_val + fp_val + fn_val)) if (2 * tp_val + fp_val + fn_val) > 0 else 0.0
        denom_f2 = (4 * prec + rec)
        f2_val = float((5 * prec * rec) / denom_f2) if denom_f2 > 0 else 0.0

        # Soft (Probability-Weighted) Confusion Counts & Soft-F2 (Training Reward Construct Only)
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)
        soft_tp_val = float(np.sum(preds_clipped[pos_mask])) if np.any(pos_mask) else 0.0
        soft_fn_val = float(np.sum(1.0 - preds_clipped[pos_mask])) if np.any(pos_mask) else 0.0
        soft_fp_val = float(np.sum(preds_clipped[neg_mask])) if np.any(neg_mask) else 0.0
        soft_tn_val = float(np.sum(1.0 - preds_clipped[neg_mask])) if np.any(neg_mask) else 0.0

        denom_soft_prec = soft_tp_val + soft_fp_val + eps
        denom_soft_rec = soft_tp_val + soft_fn_val + eps
        soft_prec = float(soft_tp_val / denom_soft_prec)
        soft_rec = float(soft_tp_val / denom_soft_rec)
        denom_soft_f2 = (4 * soft_prec + soft_rec)
        soft_f2_val = float((5 * soft_prec * soft_rec) / denom_soft_f2) if denom_soft_f2 > 0 else 0.0

        metrics_dict = {
            "loss": loss_val,
            "accuracy": f2_val,
            "precision": prec,
            "recall": rec,
            "f1": f1_val,
            "f2": f2_val,
            "roc_auc": auc_val,
            "tp": tp_val,
            "fp": fp_val,
            "tn": tn_val,
            "fn": fn_val,
            # Training-reward soft confusion constructs (not clinical/paper reporting metrics)
            "soft_tp": soft_tp_val,
            "soft_fp": soft_fp_val,
            "soft_tn": soft_tn_val,
            "soft_fn": soft_fn_val,
            "soft_precision": soft_prec,
            "soft_recall": soft_rec,
            "soft_f2": soft_f2_val
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