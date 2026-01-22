"""
=============================================================================
Script de Treinamento - Classificador de Movimento
=============================================================================
Treina o modelo de classificação de qualidade de movimento usando
dados coletados ou sintéticos.

Uso:
    # Treinar com dados sintéticos
    python -m src.train --synthetic --samples 5000

    # Treinar com dataset existente
    python -m src.train --dataset data/squat_dataset.csv

    # Treinar e converter para TFLite
    python -m src.train --synthetic --tflite

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict

# Adicionar path do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_and_prepare_data(
    dataset_path: str = None,
    exercise_type: str = 'squat',
    synthetic_samples: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Carrega e prepara os dados para treinamento.

    Args:
        dataset_path: Caminho do dataset CSV
        exercise_type: Tipo de exercício
        synthetic_samples: Número de amostras sintéticas (0 = usar dataset)

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    if synthetic_samples > 0:
        print(f"Gerando {synthetic_samples} amostras sintéticas...")
        from src.data.dataset_collector import generate_synthetic_dataset

        # Gerar dataset sintético
        output_file = f'data/synthetic_{exercise_type}.csv'
        df = generate_synthetic_dataset(exercise_type, synthetic_samples, output_file)
    else:
        print(f"Carregando dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)

    print(f"Total de amostras: {len(df)}")
    print(f"Distribuição de labels:\n{df['label'].value_counts()}")

    # Selecionar features
    feature_columns = [col for col in df.columns if col not in
                      ['label', 'timestamp', 'exercise', 'source']]

    X = df[feature_columns].values.astype(np.float32)
    y = df['label'].values.astype(np.int32)

    # Tratar NaN
    X = np.nan_to_num(X, nan=0.0)

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Treino: {len(X_train)} amostras")
    print(f"Teste: {len(X_test)} amostras")
    print(f"Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32
) -> Tuple:
    """
    Treina o modelo de classificação.

    Args:
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        epochs: Número de épocas
        batch_size: Tamanho do batch

    Returns:
        modelo, histórico, métricas
    """
    from src.models.movement_classifier import MovementClassifier, ModelConfig

    print("\n" + "="*60)
    print("TREINAMENTO DO MODELO")
    print("="*60)

    # Configurar modelo
    config = ModelConfig(
        input_features=X_train.shape[1],
        num_classes=3,
        hidden_units=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001
    )

    # Criar e construir modelo
    classifier = MovementClassifier(config)
    classifier.build()
    classifier.summary()

    # Treinar
    print("\nIniciando treinamento...")
    history = classifier.train(
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Avaliar
    print("\n" + "="*60)
    print("AVALIAÇÃO DO MODELO")
    print("="*60)

    metrics = classifier.evaluate(X_test, y_test)

    print(f"\nAcurácia: {metrics['accuracy']*100:.2f}%")
    print(f"Loss: {metrics['loss']:.4f}")

    print("\nMatriz de Confusão:")
    print(metrics['confusion_matrix'])

    print("\nRelatório de Classificação:")
    for class_name, class_metrics in metrics['classification_report'].items():
        if isinstance(class_metrics, dict):
            print(f"  {class_name}:")
            print(f"    Precision: {class_metrics['precision']:.2f}")
            print(f"    Recall: {class_metrics['recall']:.2f}")
            print(f"    F1-Score: {class_metrics['f1-score']:.2f}")

    return classifier, history, metrics


def save_model(
    classifier,
    scaler,
    output_dir: str = 'models/saved',
    convert_tflite: bool = True
):
    """
    Salva o modelo e scaler.

    Args:
        classifier: Modelo treinado
        scaler: Scaler de normalização
        output_dir: Diretório de saída
        convert_tflite: Se deve converter para TFLite
    """
    os.makedirs(output_dir, exist_ok=True)

    # Salvar modelo Keras
    keras_path = os.path.join(output_dir, 'movement_classifier.h5')
    classifier.save(keras_path)

    # Salvar scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler salvo: {scaler_path}")

    # Converter para TFLite
    if convert_tflite:
        tflite_path = os.path.join(output_dir, 'movement_classifier.tflite')
        classifier.convert_to_tflite(tflite_path, quantize=True)

    print(f"\nModelos salvos em: {output_dir}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Treinamento do Classificador de Movimento'
    )

    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Caminho do dataset CSV')
    parser.add_argument('--exercise', '-e', type=str, default='squat',
                       choices=['squat', 'pushup', 'running', 'general'],
                       help='Tipo de exercício')
    parser.add_argument('--synthetic', '-s', action='store_true',
                       help='Usar dados sintéticos')
    parser.add_argument('--samples', '-n', type=int, default=5000,
                       help='Número de amostras sintéticas')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Tamanho do batch')
    parser.add_argument('--output', '-o', type=str, default='models/saved',
                       help='Diretório de saída')
    parser.add_argument('--tflite', action='store_true',
                       help='Converter para TFLite')

    args = parser.parse_args()

    print("="*60)
    print("TREINAMENTO - CLASSIFICADOR DE MOVIMENTO ESPORTIVO")
    print("="*60)
    print(f"Exercício: {args.exercise}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # Verificar fonte de dados
    if args.synthetic:
        print(f"Fonte: Dados sintéticos ({args.samples} amostras)")
        dataset_path = None
        synthetic_samples = args.samples
    elif args.dataset:
        if not os.path.exists(args.dataset):
            print(f"Erro: Dataset não encontrado: {args.dataset}")
            sys.exit(1)
        print(f"Fonte: {args.dataset}")
        dataset_path = args.dataset
        synthetic_samples = 0
    else:
        print("Erro: Especifique --dataset ou --synthetic")
        sys.exit(1)

    # Carregar dados
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data(
        dataset_path=dataset_path,
        exercise_type=args.exercise,
        synthetic_samples=synthetic_samples
    )

    # Treinar modelo
    classifier, history, metrics = train_model(
        X_train, y_train,
        X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Salvar
    save_model(classifier, scaler, args.output, args.tflite)

    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO!")
    print("="*60)
    print(f"Acurácia final: {metrics['accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
