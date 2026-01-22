"""
=============================================================================
Módulo de Análise em Tempo Real
=============================================================================
Este módulo implementa a análise de movimento em tempo real,
integrando detecção de pose, análise biomecânica, classificador ML e feedback.

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import cv2
import numpy as np
import time
from typing import Optional, Dict, Callable
from enum import Enum
from collections import deque
import joblib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pose_detector import PoseDetector, PoseResult, PoseLandmark
from analysis.biomechanics import BiomechanicsAnalyzer, ExerciseAnalyzer, JointType
from analysis.feedback_generator import FeedbackGenerator, FeedbackMessage, FeedbackLevel

# Tentar importar classificador TFLite
try:
    from models.movement_classifier import TFLiteClassifier, prepare_features
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False


class ExerciseType(Enum):
    """Tipos de exercícios suportados."""
    SQUAT = "squat"
    PUSHUP = "pushup"
    RUNNING = "running"
    JUMPING_JACK = "jumping_jack"
    GENERAL = "general"


class SportsMotionAnalyzer:
    """
    Analisador de movimento esportivo em tempo real.
    
    Integra detecção de pose, análise biomecânica e feedback
    para fornecer análise completa em tempo real.
    
    Exemplo de Uso:
        >>> analyzer = SportsMotionAnalyzer(exercise_type='squat')
        >>> analyzer.start_realtime_analysis()
    """
    
    def __init__(
        self,
        exercise_type: str = 'general',
        camera_index: int = 0,
        video_path: Optional[str] = None,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        use_ml_classifier: bool = True,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ):
        """
        Inicializa o analisador.

        Args:
            exercise_type: Tipo de exercício ('squat', 'pushup', 'general')
            camera_index: Índice da câmera
            video_path: Caminho para vídeo (None para câmera ao vivo)
            model_complexity: Complexidade do modelo (0, 1, 2)
            min_detection_confidence: Confiança mínima de detecção
            use_ml_classifier: Se deve usar o classificador ML (TFLite)
            model_path: Caminho para o modelo TFLite
            scaler_path: Caminho para o scaler
        """
        self.exercise_type = ExerciseType(exercise_type)
        self.camera_index = camera_index
        self.video_path = video_path

        # Inicializar componentes
        self.pose_detector = PoseDetector(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )
        self.exercise_analyzer = ExerciseAnalyzer()
        self.feedback_generator = FeedbackGenerator()

        # Classificador ML
        self.use_ml_classifier = use_ml_classifier and TFLITE_AVAILABLE
        self.ml_classifier = None
        self.scaler = None

        if self.use_ml_classifier:
            self._load_ml_classifier(model_path, scaler_path)

        # Estado
        self.is_running = False
        self.cap = None
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()

        # Histórico de predições para suavização
        self.prediction_history = deque(maxlen=10)

        # Callbacks
        self.on_frame_callback: Optional[Callable] = None
        self.on_analysis_callback: Optional[Callable] = None

    def _load_ml_classifier(self, model_path: Optional[str], scaler_path: Optional[str]):
        """Carrega o classificador TFLite e scaler."""
        # Caminhos padrão
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if model_path is None:
            model_path = os.path.join(base_dir, 'models', 'saved', 'movement_classifier.tflite')
        if scaler_path is None:
            scaler_path = os.path.join(base_dir, 'models', 'saved', 'scaler.joblib')

        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.ml_classifier = TFLiteClassifier(model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"Classificador ML carregado: {model_path}")
            else:
                print(f"Modelo não encontrado. Usando análise baseada em regras.")
                self.use_ml_classifier = False
        except Exception as e:
            print(f"Erro ao carregar classificador ML: {e}")
            self.use_ml_classifier = False
    
    def start_realtime_analysis(self):
        """
        Inicia análise em tempo real.
        
        Abre câmera ou vídeo e processa frames continuamente.
        Pressione 'q' para sair, 'r' para resetar contador.
        """
        # Abrir fonte de vídeo
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir a fonte de vídeo")
            return
        
        # Configurar resolução
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.is_running = True
        print(f"Iniciando análise de {self.exercise_type.value}...")
        print("Controles: 'q'=sair, 'r'=resetar, 's'=screenshot")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                if self.video_path:
                    # Loop do vídeo
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            # Processar frame
            processed_frame, analysis_result = self.process_frame(frame)
            
            # Calcular FPS
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time + 1e-6)
            self.fps_history.append(fps)
            self.last_time = current_time
            avg_fps = np.mean(list(self.fps_history))
            
            # Adicionar FPS ao frame
            cv2.putText(
                processed_frame,
                f"FPS: {avg_fps:.1f}",
                (processed_frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Mostrar frame
            cv2.imshow("Sports Motion Analyzer", processed_frame)
            
            # Callbacks
            if self.on_frame_callback:
                self.on_frame_callback(processed_frame)
            if self.on_analysis_callback and analysis_result:
                self.on_analysis_callback(analysis_result)
            
            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.exercise_analyzer.reset()
                print("Contador resetado!")
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Screenshot salvo: {filename}")
        
        self.stop()
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Processa um único frame.
        
        Args:
            frame: Frame de vídeo
        
        Returns:
            Tuple (frame_processado, resultado_analise)
        """
        # Detectar pose
        pose_result = self.pose_detector.detect(frame)
        
        if not pose_result:
            # Sem pose detectada
            cv2.putText(
                frame,
                "Nenhuma pessoa detectada",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return frame, None
        
        # Desenhar pose
        frame = self.pose_detector.draw_landmarks(frame, pose_result)
        
        # Analisar exercício
        analysis_result = self._analyze_exercise(pose_result)
        
        # Gerar feedback
        feedback = self._generate_feedback(analysis_result)
        
        # Desenhar feedback
        frame = self.feedback_generator.draw_feedback_on_frame(frame, feedback)
        
        # Desenhar métricas específicas do exercício
        frame = self._draw_exercise_specific(frame, pose_result, analysis_result)
        
        return frame, analysis_result
    
    def _analyze_exercise(self, pose_result: PoseResult) -> Dict:
        """Analisa o exercício baseado no tipo."""
        if self.exercise_type == ExerciseType.SQUAT:
            result = self.exercise_analyzer.analyze_squat(pose_result)
        elif self.exercise_type == ExerciseType.PUSHUP:
            result = self.exercise_analyzer.analyze_pushup(pose_result)
        else:
            # Análise geral
            bio_result = self.exercise_analyzer.biomechanics.analyze(pose_result)
            result = {
                'biomechanics': bio_result,
                'symmetry': bio_result.symmetry_score,
                'body_alignment': bio_result.body_alignment,
            }

        # Classificação ML se disponível
        if self.use_ml_classifier and self.ml_classifier:
            ml_result = self._classify_with_ml(pose_result)
            result['ml_classification'] = ml_result
            # Atualizar quality_score com predição ML
            if ml_result:
                result['quality_score'] = ml_result['quality_score']
                result['quality_label'] = ml_result['quality_label']
                result['quality_confidence'] = ml_result['confidence']

        return result

    def _classify_with_ml(self, pose_result: PoseResult) -> Optional[Dict]:
        """
        Classifica a qualidade do movimento usando o modelo ML.

        Args:
            pose_result: Resultado da detecção de pose

        Returns:
            Dicionário com classificação ou None
        """
        if not self.ml_classifier or not self.scaler:
            return None

        try:
            # Extrair features dos landmarks
            features = self._extract_features(pose_result)

            if features is None:
                return None

            # Normalizar com scaler
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Predição
            probs = self.ml_classifier.predict_proba(features_scaled)[0]
            predicted_class = np.argmax(probs)
            confidence = probs[predicted_class]

            # Adicionar ao histórico para suavização
            self.prediction_history.append(predicted_class)

            # Classe suavizada (votação majoritária)
            if len(self.prediction_history) >= 3:
                smoothed_class = int(np.median(list(self.prediction_history)))
            else:
                smoothed_class = predicted_class

            # Labels
            quality_labels = {0: 'Ruim', 1: 'Médio', 2: 'Bom'}

            return {
                'predicted_class': int(smoothed_class),
                'quality_label': quality_labels.get(smoothed_class, 'Desconhecido'),
                'quality_score': smoothed_class / 2.0,  # 0.0, 0.5, 1.0
                'confidence': float(confidence),
                'probabilities': {
                    'ruim': float(probs[0]),
                    'medio': float(probs[1]),
                    'bom': float(probs[2])
                }
            }
        except Exception as e:
            print(f"Erro na classificação ML: {e}")
            return None

    def _extract_features(self, pose_result: PoseResult) -> Optional[np.ndarray]:
        """
        Extrai features dos landmarks para o classificador.

        Args:
            pose_result: Resultado da detecção de pose

        Returns:
            Array de features ou None
        """
        try:
            features = []

            # Landmarks relevantes (13 pontos)
            landmark_indices = [
                PoseLandmark.NOSE,
                PoseLandmark.LEFT_SHOULDER,
                PoseLandmark.RIGHT_SHOULDER,
                PoseLandmark.LEFT_ELBOW,
                PoseLandmark.RIGHT_ELBOW,
                PoseLandmark.LEFT_WRIST,
                PoseLandmark.RIGHT_WRIST,
                PoseLandmark.LEFT_HIP,
                PoseLandmark.RIGHT_HIP,
                PoseLandmark.LEFT_KNEE,
                PoseLandmark.RIGHT_KNEE,
                PoseLandmark.LEFT_ANKLE,
                PoseLandmark.RIGHT_ANKLE
            ]

            # Extrair x, y, z, visibility para cada landmark
            for idx in landmark_indices:
                lm = pose_result.get_landmark(idx)
                if lm:
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])

            # Calcular ângulos articulares
            angles = self._calculate_joint_angles(pose_result)
            for angle_name in ['left_knee', 'right_knee', 'left_hip', 'right_hip',
                               'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder']:
                angle_val = angles.get(angle_name, 0.0)
                features.append(angle_val / 180.0)  # Normalizar para 0-1

            return np.array(features, dtype=np.float32)

        except Exception as e:
            print(f"Erro ao extrair features: {e}")
            return None

    def _calculate_joint_angles(self, pose_result: PoseResult) -> Dict[str, float]:
        """Calcula ângulos das articulações."""
        from preprocessing.pose_detector import calculate_angle

        angles = {}
        w, h = pose_result.frame_width, pose_result.frame_height

        # Definição das articulações (ponto1, vértice, ponto2)
        joint_definitions = {
            'left_knee': (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
            'right_knee': (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
            'left_hip': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
            'right_hip': (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
            'left_elbow': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
            'right_elbow': (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
            'left_shoulder': (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
            'right_shoulder': (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
        }

        for joint_name, (idx1, idx2, idx3) in joint_definitions.items():
            lm1 = pose_result.get_landmark(idx1)
            lm2 = pose_result.get_landmark(idx2)
            lm3 = pose_result.get_landmark(idx3)

            if lm1 and lm2 and lm3:
                if lm1.visibility > 0.5 and lm2.visibility > 0.5 and lm3.visibility > 0.5:
                    p1 = (lm1.x * w, lm1.y * h)
                    p2 = (lm2.x * w, lm2.y * h)
                    p3 = (lm3.x * w, lm3.y * h)
                    angles[joint_name] = calculate_angle(p1, p2, p3)
                else:
                    angles[joint_name] = 0.0
            else:
                angles[joint_name] = 0.0

        return angles
    
    def _generate_feedback(self, analysis_result: Dict) -> list:
        """Gera feedback baseado na análise."""
        if self.exercise_type == ExerciseType.SQUAT:
            return self.feedback_generator.generate_squat_feedback(analysis_result)
        elif self.exercise_type == ExerciseType.PUSHUP:
            return self.feedback_generator.generate_pushup_feedback(analysis_result)
        else:
            return []
    
    def _draw_exercise_specific(
        self, 
        frame: np.ndarray, 
        pose_result: PoseResult,
        analysis_result: Dict
    ) -> np.ndarray:
        """Desenha visualizações específicas do exercício."""
        h, w = frame.shape[:2]
        
        if self.exercise_type == ExerciseType.SQUAT:
            # Desenhar contador de repetições
            rep_count = analysis_result.get('rep_count', 0)
            frame = self.feedback_generator.draw_rep_counter(frame, rep_count)
            
            # Desenhar barra de qualidade
            quality = analysis_result.get('quality_score', 0)
            frame = self.feedback_generator.draw_quality_bar(frame, quality, (10, 80))
            
            # Desenhar ângulos dos joelhos
            bio = analysis_result.get('biomechanics')
            if bio:
                # Joelho esquerdo
                left_knee = bio.joint_angles.get('left_knee')
                if left_knee:
                    hip = pose_result.get_landmark_pixel(PoseLandmark.LEFT_HIP)
                    knee = pose_result.get_landmark_pixel(PoseLandmark.LEFT_KNEE)
                    ankle = pose_result.get_landmark_pixel(PoseLandmark.LEFT_ANKLE)
                    
                    if all([hip, knee, ankle]):
                        frame = self.feedback_generator.draw_angle_visualization(
                            frame, hip, knee, ankle, 
                            left_knee.angle, 
                            color=(255, 255, 0),
                            label="Joelho E"
                        )
                
                # Joelho direito
                right_knee = bio.joint_angles.get('right_knee')
                if right_knee:
                    hip = pose_result.get_landmark_pixel(PoseLandmark.RIGHT_HIP)
                    knee = pose_result.get_landmark_pixel(PoseLandmark.RIGHT_KNEE)
                    ankle = pose_result.get_landmark_pixel(PoseLandmark.RIGHT_ANKLE)
                    
                    if all([hip, knee, ankle]):
                        frame = self.feedback_generator.draw_angle_visualization(
                            frame, hip, knee, ankle,
                            right_knee.angle,
                            color=(0, 255, 255),
                            label="Joelho D"
                        )
            
            # Mostrar profundidade
            depth = analysis_result.get('depth', 'standing')
            depth_text = {
                'standing': 'Em pé',
                'partial': 'Parcial',
                'parallel': 'Paralelo',
                'deep': 'Profundo'
            }.get(depth, depth)
            
            cv2.putText(
                frame,
                f"Profundidade: {depth_text}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        elif self.exercise_type == ExerciseType.PUSHUP:
            # Desenhar barra de qualidade
            quality = analysis_result.get('quality_score', 0)
            frame = self.feedback_generator.draw_quality_bar(frame, quality)
            
            # Desenhar fase
            phase = analysis_result.get('phase', 'up')
            phase_text = "Subindo" if phase == 'up' else "Descendo"
            cv2.putText(
                frame,
                f"Fase: {phase_text}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Desenhar métricas comuns
        symmetry = analysis_result.get('symmetry', 0)
        alignment = analysis_result.get('body_alignment', 0)
        
        cv2.putText(
            frame,
            f"Simetria: {symmetry:.0%}",
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            frame,
            f"Alinhamento: {alignment:.1f}°",
            (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Mostrar classificação ML se disponível
        ml_result = analysis_result.get('ml_classification')
        if ml_result:
            quality_label = ml_result.get('quality_label', 'N/A')
            confidence = ml_result.get('confidence', 0)

            # Cor baseada na qualidade
            quality_colors = {
                'Ruim': (0, 0, 255),    # Vermelho
                'Médio': (0, 165, 255),  # Laranja
                'Bom': (0, 255, 0)       # Verde
            }
            color = quality_colors.get(quality_label, (255, 255, 255))

            # Desenhar caixa de qualidade ML
            cv2.rectangle(frame, (w - 200, 50), (w - 10, 130), (0, 0, 0), -1)
            cv2.rectangle(frame, (w - 200, 50), (w - 10, 130), color, 2)

            cv2.putText(
                frame,
                "Qualidade ML:",
                (w - 190, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                quality_label,
                (w - 190, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Conf: {confidence:.0%}",
                (w - 190, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1
            )

        return frame
    
    def stop(self):
        """Para a análise e libera recursos."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose_detector.release()
        print("Análise finalizada.")
    
    def set_exercise_type(self, exercise_type: str):
        """
        Muda o tipo de exercício.
        
        Args:
            exercise_type: Novo tipo de exercício
        """
        self.exercise_type = ExerciseType(exercise_type)
        self.exercise_analyzer.reset()
        print(f"Exercício alterado para: {exercise_type}")


def run_cli():
    """Executa análise via linha de comando."""
    import argparse

    parser = argparse.ArgumentParser(description='Analisador de Movimento Esportivo')
    parser.add_argument(
        '--exercise', '-e',
        type=str,
        default='squat',
        choices=['squat', 'pushup', 'general'],
        help='Tipo de exercício'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Índice da câmera'
    )
    parser.add_argument(
        '--video', '-v',
        type=str,
        default=None,
        help='Caminho para arquivo de vídeo'
    )
    parser.add_argument(
        '--model-complexity', '-m',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Complexidade do modelo (0=Lite, 1=Full, 2=Heavy)'
    )
    parser.add_argument(
        '--no-ml',
        action='store_true',
        help='Desabilita classificador ML (usa apenas regras)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Caminho para modelo TFLite personalizado'
    )

    args = parser.parse_args()

    # Criar analisador
    analyzer = SportsMotionAnalyzer(
        exercise_type=args.exercise,
        camera_index=args.camera,
        video_path=args.video,
        model_complexity=args.model_complexity,
        use_ml_classifier=not args.no_ml,
        model_path=args.model_path
    )

    # Iniciar análise
    analyzer.start_realtime_analysis()


if __name__ == "__main__":
    run_cli()
