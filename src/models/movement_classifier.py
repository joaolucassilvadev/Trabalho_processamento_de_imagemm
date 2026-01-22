"""
=============================================================================
Modelo de Classificação de Qualidade de Movimento
=============================================================================
Rede neural para classificar a qualidade de movimentos esportivos
baseado em landmarks do MediaPipe.

Classes:
    - 0: Movimento incorreto/ruim
    - 1: Movimento parcialmente correto
    - 2: Movimento correto/bom

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import numpy as np
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


@dataclass
class ModelConfig:
    """Configuração do modelo."""
    input_features: int = 60  # landmarks (13*4) + ângulos (8)
    num_classes: int = 3
    hidden_units: List[int] = None
    dropout_rate: float = 0.3
    learning_rate: float = 0.001

    def __post_init__(self):
        if self.hidden_units is None:
            self.hidden_units = [128, 64, 32]


class MovementClassifier:
    """
    Classificador de qualidade de movimento usando rede neural.

    Arquitetura:
        Input (features) → Dense layers → Softmax → Classes (3)

    Exemplo de Uso:
        >>> classifier = MovementClassifier()
        >>> classifier.build()
        >>> classifier.train(X_train, y_train, X_val, y_val)
        >>> predictions = classifier.predict(X_test)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Inicializa o classificador.

        Args:
            config: Configuração do modelo
        """
        self.config = config or ModelConfig()
        self.model: Optional[Model] = None
        self.history = None

    def build(self) -> Model:
        """
        Constrói a arquitetura do modelo.

        Returns:
            Modelo Keras compilado
        """
        inputs = keras.Input(shape=(self.config.input_features,), name='input')

        x = inputs

        # Normalização da entrada
        x = layers.BatchNormalization()(x)

        # Camadas densas
        for i, units in enumerate(self.config.hidden_units):
            x = layers.Dense(units, name=f'dense_{i}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(self.config.dropout_rate)(x)

        # Camada de saída
        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            name='output'
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='MovementClassifier')

        # Compilar
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def build_lstm(self, sequence_length: int = 30) -> Model:
        """
        Constrói modelo LSTM para análise temporal.

        Args:
            sequence_length: Número de frames na sequência

        Returns:
            Modelo Keras compilado
        """
        inputs = keras.Input(
            shape=(sequence_length, self.config.input_features),
            name='input'
        )

        x = inputs

        # LSTM layers
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(self.config.dropout_rate)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dropout(self.config.dropout_rate)(x)

        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config.dropout_rate)(x)

        outputs = layers.Dense(
            self.config.num_classes,
            activation='softmax',
            name='output'
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs, name='MovementClassifierLSTM')

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def summary(self):
        """Mostra resumo do modelo."""
        if self.model:
            self.model.summary()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict:
        """
        Treina o modelo.

        Args:
            X_train: Dados de treinamento
            y_train: Labels de treinamento
            X_val: Dados de validação
            y_val: Labels de validação
            epochs: Número de épocas
            batch_size: Tamanho do batch
            verbose: Nível de verbosidade

        Returns:
            Histórico de treinamento
        """
        if self.model is None:
            self.build()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Treinar
        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições.

        Args:
            X: Dados de entrada

        Returns:
            Classes preditas
        """
        if self.model is None:
            raise ValueError("Modelo não treinado")

        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidades de cada classe.

        Args:
            X: Dados de entrada

        Returns:
            Probabilidades para cada classe
        """
        if self.model is None:
            raise ValueError("Modelo não treinado")

        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Avalia o modelo.

        Args:
            X_test: Dados de teste
            y_test: Labels de teste

        Returns:
            Métricas de avaliação
        """
        if self.model is None:
            raise ValueError("Modelo não treinado")

        # Métricas básicas
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        # Predições
        y_pred = self.predict(X_test)

        # Matriz de confusão
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test, y_pred)

        # Report
        report = classification_report(
            y_test, y_pred,
            target_names=['Ruim', 'Médio', 'Bom'],
            output_dict=True
        )

        return {
            'loss': loss,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }

    def save(self, filepath: str):
        """
        Salva o modelo.

        Args:
            filepath: Caminho do arquivo
        """
        if self.model is None:
            raise ValueError("Modelo não treinado")

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.model.save(filepath)
        print(f"Modelo salvo: {filepath}")

    def load(self, filepath: str):
        """
        Carrega o modelo.

        Args:
            filepath: Caminho do arquivo
        """
        self.model = keras.models.load_model(filepath)
        print(f"Modelo carregado: {filepath}")

    def convert_to_tflite(
        self,
        output_path: str,
        quantize: bool = True
    ) -> str:
        """
        Converte modelo para TensorFlow Lite.

        Args:
            output_path: Caminho de saída
            quantize: Se deve aplicar quantização

        Returns:
            Caminho do modelo TFLite
        """
        if self.model is None:
            raise ValueError("Modelo não treinado")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        print(f"Modelo TFLite salvo: {output_path} ({size_kb:.2f} KB)")

        return output_path


class TFLiteClassifier:
    """
    Classificador usando modelo TensorFlow Lite.

    Para inferência em tempo real com baixa latência.
    """

    def __init__(self, model_path: str):
        """
        Inicializa o classificador TFLite.

        Args:
            model_path: Caminho do modelo .tflite
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições usando TFLite.

        Args:
            X: Dados de entrada

        Returns:
            Classes preditas
        """
        # Garantir formato correto
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X = X.astype(self.input_dtype)

        predictions = []
        for sample in X:
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                sample.reshape(1, -1)
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(np.argmax(output[0]))

        return np.array(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidades.

        Args:
            X: Dados de entrada

        Returns:
            Probabilidades
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X = X.astype(self.input_dtype)

        probabilities = []
        for sample in X:
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                sample.reshape(1, -1)
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            probabilities.append(output[0])

        return np.array(probabilities)


def prepare_features(landmarks_dict: Dict, angles_dict: Dict) -> np.ndarray:
    """
    Prepara features a partir de landmarks e ângulos.

    Args:
        landmarks_dict: Dicionário com coordenadas dos landmarks
        angles_dict: Dicionário com ângulos articulares

    Returns:
        Array de features normalizado
    """
    features = []

    # Landmarks (x, y, z, visibility)
    landmark_names = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                     'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    for lm in landmark_names:
        features.append(landmarks_dict.get(f'{lm}_x', 0.0))
        features.append(landmarks_dict.get(f'{lm}_y', 0.0))
        features.append(landmarks_dict.get(f'{lm}_z', 0.0))
        features.append(landmarks_dict.get(f'{lm}_vis', 0.0))

    # Ângulos
    angle_names = ['left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle',
                  'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle']

    for angle in angle_names:
        # Normalizar ângulos para 0-1
        features.append(angles_dict.get(angle, 0.0) / 180.0)

    return np.array(features, dtype=np.float32)


if __name__ == '__main__':
    # Teste do módulo
    print("Testando MovementClassifier...")

    # Criar modelo
    config = ModelConfig(input_features=60, num_classes=3)
    classifier = MovementClassifier(config)
    classifier.build()
    classifier.summary()

    # Dados de teste sintéticos
    X_test = np.random.randn(100, 60).astype(np.float32)
    y_test = np.random.randint(0, 3, 100)

    # Treinar brevemente
    history = classifier.train(X_test, y_test, epochs=5, verbose=1)

    # Predição
    predictions = classifier.predict(X_test[:5])
    print(f"Predições: {predictions}")

    print("Teste concluído!")
