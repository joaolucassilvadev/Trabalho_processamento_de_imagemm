"""
=============================================================================
Aplicação Web - Analisador de Movimento Esportivo
=============================================================================
Servidor Flask que serve a aplicação web com MediaPipe.js para análise
de movimento no navegador. Inclui API para classificação ML.

Uso:
    python -m app.web_app
    Acesse: http://localhost:5000

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import os
import sys
import numpy as np
from flask import Flask, render_template, send_from_directory, request, jsonify
import joblib

# Adicionar path do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tentar importar classificador TFLite
try:
    from src.models.movement_classifier import TFLiteClassifier
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Classificador ML global
ml_classifier = None
scaler = None

# Carregar modelo na inicialização (para Gunicorn)
with app.app_context():
    pass  # Será carregado abaixo


def load_ml_model():
    """Carrega o modelo TFLite e scaler."""
    global ml_classifier, scaler

    if not TFLITE_AVAILABLE:
        print("TensorFlow Lite não disponível")
        return False

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'saved', 'movement_classifier.tflite')
    scaler_path = os.path.join(base_dir, 'models', 'saved', 'scaler.joblib')

    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            ml_classifier = TFLiteClassifier(model_path)
            scaler = joblib.load(scaler_path)
            print(f"Modelo ML carregado: {model_path}")
            return True
        else:
            print(f"Modelo não encontrado em: {model_path}")
            return False
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return False


@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve arquivos estáticos."""
    return send_from_directory(app.static_folder, filename)


@app.route('/api/classify', methods=['POST'])
def classify_movement():
    """
    API para classificar qualidade do movimento.

    Recebe landmarks e ângulos do frontend e retorna classificação ML.

    Body JSON esperado:
    {
        "landmarks": {
            "nose_x": 0.5, "nose_y": 0.3, ...
        },
        "angles": {
            "left_knee_angle": 90, ...
        }
    }
    """
    global ml_classifier, scaler

    if not ml_classifier or not scaler:
        return jsonify({
            'success': False,
            'error': 'Modelo não carregado',
            'quality_label': 'N/A'
        })

    try:
        data = request.get_json()
        landmarks = data.get('landmarks', {})
        angles = data.get('angles', {})

        # Extrair features
        features = extract_features_from_json(landmarks, angles)

        if features is None:
            return jsonify({
                'success': False,
                'error': 'Features inválidas',
                'quality_label': 'N/A'
            })

        # Normalizar
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predição
        probs = ml_classifier.predict_proba(features_scaled)[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        # Labels
        quality_labels = {0: 'Ruim', 1: 'Médio', 2: 'Bom'}

        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'quality_label': quality_labels.get(predicted_class, 'Desconhecido'),
            'quality_score': predicted_class / 2.0,
            'confidence': confidence,
            'probabilities': {
                'ruim': float(probs[0]),
                'medio': float(probs[1]),
                'bom': float(probs[2])
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'quality_label': 'Erro'
        })


@app.route('/api/status')
def api_status():
    """Retorna status da API e modelo."""
    return jsonify({
        'status': 'online',
        'ml_available': ml_classifier is not None,
        'tflite_available': TFLITE_AVAILABLE
    })


def extract_features_from_json(landmarks: dict, angles: dict) -> np.ndarray:
    """
    Extrai features dos landmarks e ângulos recebidos via JSON.

    Args:
        landmarks: Dicionário com coordenadas dos landmarks
        angles: Dicionário com ângulos articulares

    Returns:
        Array de features
    """
    features = []

    # Landmarks (13 pontos x 4 valores)
    landmark_names = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                      'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    for lm in landmark_names:
        features.append(landmarks.get(f'{lm}_x', 0.0))
        features.append(landmarks.get(f'{lm}_y', 0.0))
        features.append(landmarks.get(f'{lm}_z', 0.0))
        features.append(landmarks.get(f'{lm}_visibility', 1.0))

    # Ângulos (8 valores)
    angle_names = ['left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle',
                   'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle']

    for angle in angle_names:
        # Normalizar ângulos para 0-1
        angle_val = angles.get(angle, 0.0)
        features.append(angle_val / 180.0)

    return np.array(features, dtype=np.float32)


def main():
    """Função principal."""
    # Criar diretórios se não existirem
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    # Carregar modelo ML
    load_ml_model()

    print("="*60)
    print("   ANALISADOR DE MOVIMENTO ESPORTIVO - Web")
    print("="*60)
    print(f"   ML Classifier: {'Ativo' if ml_classifier else 'Inativo'}")
    print("   Acesse: http://localhost:5000")
    print("="*60)

    app.run(host='0.0.0.0', port=5000, debug=True)


# Carregar modelo ao importar (para Gunicorn/Render)
load_ml_model()

if __name__ == '__main__':
    main()
