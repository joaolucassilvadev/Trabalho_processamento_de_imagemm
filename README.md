# Analisador de Movimento Esportivo

Sistema de análise biomecânica em tempo real usando visão computacional e Machine Learning.

## Funcionalidades

- Detecção de pose corporal em tempo real (MediaPipe)
- Análise biomecânica de exercícios (agachamento e flexão)
- Classificação de qualidade do movimento com ML (TensorFlow Lite)
- Contador automático de repetições
- Feedback visual em tempo real
- Interface web responsiva

## Tecnologias

- **Frontend**: HTML5, CSS3, JavaScript, MediaPipe.js
- **Backend**: Python, Flask
- **ML**: TensorFlow Lite, scikit-learn
- **Deploy**: Render

## Como Usar

### Acesso Online
Acesse a aplicação em: [URL do Render após deploy]

### Execução Local

```bash
# Clonar repositório
git clone https://github.com/joaolucassilvadev/Trabalho_processamento_de_imagemm.git
cd Trabalho_processamento_de_imagemm

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar
python -m app.web_app
```

Acesse: http://localhost:5000

## Estrutura do Projeto

```
├── app/
│   ├── web_app.py          # Servidor Flask
│   └── templates/
│       └── index.html      # Interface web
├── src/
│   ├── models/
│   │   └── movement_classifier.py  # Classificador ML
│   └── ...
├── models/saved/
│   ├── movement_classifier.tflite  # Modelo TFLite
│   └── scaler.joblib               # Normalizador
├── requirements.txt
├── render.yaml
└── README.md
```

## API

### GET /api/status
Retorna status da API e modelo ML.

### POST /api/classify
Classifica qualidade do movimento.

**Body:**
```json
{
  "landmarks": {"nose_x": 0.5, "nose_y": 0.3, ...},
  "angles": {"left_knee_angle": 90, ...}
}
```

**Response:**
```json
{
  "success": true,
  "quality_label": "Bom",
  "confidence": 0.95,
  "probabilities": {"ruim": 0.02, "medio": 0.03, "bom": 0.95}
}
```

## Autor

Projeto Acadêmico - Processamento de Imagem
