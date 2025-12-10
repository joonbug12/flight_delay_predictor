import tensorflow as tf
from tensorflow.keras import layers, Model, models

class FlightDelayModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()
        
    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.input_dim,))
        
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        cls_output = layers.Dense(1, activation='sigmoid', name='classification')(x)
        reg_output = layers.Dense(1, activation='linear', name='regression')(x)
        
        model = Model(inputs=inputs, outputs=[cls_output, reg_output])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={'classification': 'binary_crossentropy', 'regression': 'mae'},
            loss_weights={'classification': 0.7, 'regression': 0.3},
            metrics={
                'classification': ['accuracy', tf.keras.metrics.AUC(name='auc')],
                'regression': ['mae', 'mse']
            }
        )
        
        return model
    
    def train(self, X, y_cls, y_reg, epochs=30, batch_size=256):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.model.fit(
            X,
            {'classification': y_cls, 'regression': y_reg},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
    
    def predict(self, X):
        return self.model.predict(X, verbose=0)
    
    def save(self, path='output/model.h5'):
        self.model.save(path)
        
    def load(self, path):
        self.model = models.load_model(path, compile=False)