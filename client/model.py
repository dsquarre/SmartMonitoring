import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Model:

    def __init__(self, path):

        self.data = np.load(
            path,
            mmap_mode='r'
        )
        
        self.dataset_name = path
        
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

        x = tf.keras.layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        
        x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

        x = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)

        x = tf.keras.layers.SeparableConv1D(256,3,padding='same',activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        anomaly_output = tf.keras.layers.Dense(1, activation='sigmoid', name='anomaly')(x)
        disease_output = tf.keras.layers.Dense(4, activation='softmax', name='disease')(x)

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

    def train_local_gradients_fv(self):
        """
        CUSTOM LOOP FOR FEDFV: Computes true, accumulated raw gradients 
        across the local dataset without using any local adaptive optimizer.
        """
        # Initialize gradient accumulation matrices matching trainable weights shape
        accumulated_grads = [np.zeros(var.shape, dtype=np.float32) for var in self.model.trainable_variables]
        total_loss = 0.0
        batch_count = 0

        total_steps = len(self.train_indices) // 16
        print(f"Beginning local FedFV Gradient Extraction ({total_steps} steps total)...")

        # FIX 1: Added parentheses () to call the generator function
        # FIX 2: Adjusted unpacking to match the generator's dictionary yield output structure
        for X_batch, y_batch_dict in self.train_generator(batch_size=16):
            
            # Extract target vectors from the batch payload dictionary
            y_anom = y_batch_dict["anomaly"]
            y_dise = y_batch_dict["disease"]

            with tf.GradientTape() as tape:
                # Forward pass
                predictions = self.model(X_batch, training=True)
                
                # FIX 3: Extracted dictionary values from the network's multi-output indices
                # Predictions order mirrors your build_model definition: [anomaly_output, disease_output]
                pred_anom = predictions[0]
                pred_dise = predictions[1]
                
                # FIX 4: Aligned loss calculations with your specific build_model parameters
                # Anomaly is binary classification (sigmoid) -> binary_crossentropy
                # Disease is multi-class classification (softmax, 4 categories) -> categorical_crossentropy
                loss_anom = tf.keras.losses.binary_crossentropy(y_anom, tf.squeeze(pred_anom, axis=-1))
                loss_dise = tf.keras.losses.categorical_crossentropy(y_dise, pred_dise)
                
                batch_loss = tf.reduce_mean(loss_anom + loss_dise)

            # Extract raw mathematical gradients
            raw_grads = tape.gradient(batch_loss, self.model.trainable_variables)

            # Accumulate across steps
            for i, grad in enumerate(raw_grads):
                if grad is not None:
                    accumulated_grads[i] += grad.numpy()
                    
            total_loss += float(batch_loss)
            batch_count += 1

            if batch_count % 500 == 0 or batch_count == total_steps:
                print(f"[Step {batch_count}/{total_steps}] Current Running Batch Loss: {float(batch_loss):.4f}")

        # Avoid zero division if dataset is empty
        if batch_count == 0:
            return [np.zeros(var.shape) for var in self.model.trainable_variables], 0.0

        # Calculate average global updates
        average_grads = [g / batch_count for g in accumulated_grads]
        average_loss = total_loss / batch_count

        print(f"Finished Local Generation. Average Epoch Loss: {average_loss:.4f}")

        return average_grads, average_loss

    def apply_global_gradients_fv(self, global_gradients, server_lr=0.001):
        """
        Applies conflict-resolved global updates directly to the client's local 
        trainable variables using element-wise array operations.
        """
        # Convert the decoded incoming gradients list to numpy arrays safely
        native_gradients = [np.array(gg, dtype=np.float32) for gg in global_gradients]
        
        # FIX: Iterate and modify trainable_variables directly to respect the structural layout
        for var, gg in zip(self.model.trainable_variables, native_gradients):
            # Read current parameter matrix values, subtract scaled gradient, and update variable inplace
            current_value = var.read_value().numpy()
            updated_value = current_value - (server_lr * gg)
            var.assign(updated_value)
            
        print(f"Successfully applied optimized global gradients.")

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
        

        results = self.model.evaluate(
        test_ds,
        verbose=0,
        return_dict=True
        )
        
        metrics = {
            "total_loss": results["loss"],
    
            # --- Anomaly Metrics (Binary) ---
            "anomaly_accuracy": accuracy_score(
                true_anom,
                pred_anom
            ),
            "anomaly_precision": precision_score(
                true_anom,
                pred_anom,
                zero_division=0
            ),
            "anomaly_recall": recall_score(
                true_anom,
                pred_anom,
                zero_division=0
            ),
    
            # --- Disease Metrics (Multi-class) ---
            "disease_accuracy": accuracy_score(
                true_dis,
                pred_dis
            ),
            "disease_precision": precision_score(
                true_dis,
                pred_dis,
                average='weighted',
                zero_division=0
            ),
            "disease_recall": recall_score(
                true_dis,
                pred_dis,
                average='weighted',
                zero_division=0
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