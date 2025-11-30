import numpy as np
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers, optimizers # type: ignore
from typing import List, Optional, Dict, Any
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

class TFModel:
    """Wrapper addestramento/predizione con TensorFlow/Keras"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            params: Dizionario di configurazione del modello
        """

        self.params = params or {
            "architecture": [64],
            "activation": "relu",
            "dropout": 0.0,
            "batch_size": 32,
            "learning_rate": 53e-4,
            "optimizer": "adam",
            "weight_decay": 5e-4,
            "epochs": 600,
            "early_stop_patience": 50
        }
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        self.model: keras.Model | None = None
        self.trained = False
        self.history = []

    def build(self, input_dim: int, output_dim: int = 1):
        """Costruzione dinamica della rete"""
        hidden_units: List[int] = self.params.get("architecture", [])
        activation = self.params.get("activation", "relu").lower()
        if activation == "none":
            activation = None
        dropout: List[float] = (self.params.get("dropout", [0.0]))

        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        if hidden_units == []:
            raise ValueError("La lista dei neuroni è vuota! Controlla la configurazione.")

        for i, units in enumerate(hidden_units):
            x = layers.Dense(units, activation=activation)(x)
            if i < len(dropout):
                x = layers.Dropout(dropout[i])(x)

        outputs = layers.Dense(output_dim, activation='linear')(x)
        self.model = keras.Model(inputs, outputs)
        
        if self.model:
            self.model.summary()  # Log architecture

    def get_weights(self):
        if self.model:
            return self.model.get_weights()
        
    def set_weights(self, weights):
        if self.model:
            return self.model.set_weights(weights) 
        
    def get_layer(self, layer_name: str):
        if self.model:
            return self.model.get_layer(layer_name)
        
    def train(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray, 
            batch_size: Optional[int] = None,
            learning_rate: Optional[float] = None,
            epochs: Optional[int] = None,
            val_split = 0.1, 
            early_stop: bool = True) -> None:
        """Train with optional parameter overrides."""
        assert self.model is not None, "Call build() first"

        # Override params if provided
        batch_size = batch_size or int(self.params.get("batch_size", 32))
        lr = learning_rate or float(self.params.get("learning_rate", 1e-3))
        epochs = epochs or int(self.params.get("epochs", 50))
        
        opt_name = str(self.params.get("optimizer", "adam")).lower()

        if opt_name == "sgd":
            optimizer = optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif opt_name == "adamw":
            optimizer = optimizers.AdamW(learning_rate=lr, weight_decay=self.params.get("weight_decay", 0.0))
        else:
            optimizer = optimizers.Adam(learning_rate=lr)

        self.model.compile(
            optimizer=optimizer, 
            loss='mse',
            metrics=["mae", "mse"]
        )

        if early_stop:
            with tf.device(self.device):
                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=self.params.get("early_stop_patience", 10),
                    restore_best_weights=True,
                    #verbose=1
                )

                hist = self.model.fit(
                    X_train, y_train,
                    validation_split= val_split,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    callbacks=[early_stop]
                )
                
                self.history = hist.history
        else:
            with tf.device(self.device):
                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=epochs+1,
                    restore_best_weights=True,
                    #verbose=1
                )

                hist = self.model.fit(
                    X_train, y_train,
                    validation_split= val_split,
                    callbacks=[early_stop],
                    batch_size=batch_size,
                    epochs=epochs
                )
                
                self.history = hist.history

        self.trained = True

        print("Training completed.\n")
        print(f"Final val loss: {self.history['val_loss'][-1]:.4f}")
        print(f"Final val mae: {self.history['val_mae'][-1]:.4f}")
        print(f"Final val mse: {self.history['val_mse'][-1]:.4f}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self.model is not None and self.trained, "Model not trained"
        with tf.device(self.device):
            preds = self.model.predict(X_test, verbose=0)
        return preds