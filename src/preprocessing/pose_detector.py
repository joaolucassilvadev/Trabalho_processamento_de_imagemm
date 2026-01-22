"""
=============================================================================
Módulo de Detecção de Pose Corporal
=============================================================================
Este módulo implementa a detecção de pose usando MediaPipe Pose,
extraindo 33 landmarks do corpo humano.

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List, Dict, NamedTuple
from dataclasses import dataclass
from enum import IntEnum


class PoseLandmark(IntEnum):
    """Índices dos landmarks do MediaPipe Pose."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class Landmark:
    """Representa um landmark detectado."""
    x: float  # Coordenada x normalizada (0-1)
    y: float  # Coordenada y normalizada (0-1)
    z: float  # Coordenada z (profundidade)
    visibility: float  # Confiança da detecção (0-1)
    
    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Converte coordenadas normalizadas para pixels."""
        return (int(self.x * width), int(self.y * height))
    
    def to_array(self) -> np.ndarray:
        """Retorna coordenadas como array numpy."""
        return np.array([self.x, self.y, self.z])


@dataclass
class PoseResult:
    """Resultado da detecção de pose."""
    landmarks: List[Landmark]
    timestamp: float
    frame_width: int
    frame_height: int
    
    def get_landmark(self, index: int) -> Optional[Landmark]:
        """Retorna landmark pelo índice."""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None
    
    def get_landmark_pixel(self, index: int) -> Optional[Tuple[int, int]]:
        """Retorna coordenadas em pixels do landmark."""
        landmark = self.get_landmark(index)
        if landmark:
            return landmark.to_pixel(self.frame_width, self.frame_height)
        return None
    
    def to_numpy(self) -> np.ndarray:
        """Converte todos os landmarks para array numpy (33, 4)."""
        return np.array([
            [lm.x, lm.y, lm.z, lm.visibility] 
            for lm in self.landmarks
        ])


class PoseDetector:
    """
    Detector de pose corporal usando MediaPipe.
    
    Detecta 33 landmarks do corpo humano e fornece
    métodos para visualização e análise.
    
    Exemplo de Uso:
        >>> detector = PoseDetector()
        >>> cap = cv2.VideoCapture(0)
        >>> ret, frame = cap.read()
        >>> result = detector.detect(frame)
        >>> if result:
        ...     frame = detector.draw_landmarks(frame, result)
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Inicializa o detector de pose.
        
        Args:
            static_image_mode: True para imagens estáticas, False para vídeo
            model_complexity: 0=Lite, 1=Full, 2=Heavy (mais preciso)
            smooth_landmarks: Suaviza landmarks entre frames
            enable_segmentation: Habilita máscara de segmentação
            min_detection_confidence: Confiança mínima para detecção
            min_tracking_confidence: Confiança mínima para tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self._frame_count = 0
    
    def detect(self, frame: np.ndarray) -> Optional[PoseResult]:
        """
        Detecta pose em um frame.
        
        Args:
            frame: Imagem BGR do OpenCV
        
        Returns:
            PoseResult se pose detectada, None caso contrário
        """
        # Converter BGR para RGB (MediaPipe usa RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Extrair landmarks
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append(Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            ))
        
        self._frame_count += 1
        
        return PoseResult(
            landmarks=landmarks,
            timestamp=self._frame_count / 30.0,  # Assumindo 30 FPS
            frame_width=frame.shape[1],
            frame_height=frame.shape[0]
        )
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        draw_connections: bool = True,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Desenha landmarks e conexões no frame.
        
        Args:
            frame: Imagem para desenhar
            pose_result: Resultado da detecção
            draw_connections: Se deve desenhar linhas entre landmarks
            landmark_color: Cor dos pontos (BGR)
            connection_color: Cor das linhas (BGR)
            thickness: Espessura das linhas
        
        Returns:
            Frame com landmarks desenhados
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Desenhar conexões primeiro (ficam atrás dos pontos)
        if draw_connections:
            connections = self.mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start = pose_result.get_landmark(start_idx)
                end = pose_result.get_landmark(end_idx)
                
                if start and end and start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = start.to_pixel(w, h)
                    end_point = end.to_pixel(w, h)
                    cv2.line(frame, start_point, end_point, connection_color, thickness)
        
        # Desenhar landmarks
        for i, landmark in enumerate(pose_result.landmarks):
            if landmark.visibility > 0.5:
                point = landmark.to_pixel(w, h)
                cv2.circle(frame, point, 5, landmark_color, -1)
                cv2.circle(frame, point, 7, (0, 0, 0), 1)  # Borda preta
        
        return frame
    
    def draw_landmarks_styled(
        self,
        frame: np.ndarray,
        pose_result: PoseResult
    ) -> np.ndarray:
        """
        Desenha landmarks com estilo padrão do MediaPipe.
        
        Args:
            frame: Imagem para desenhar
            pose_result: Resultado da detecção
        
        Returns:
            Frame com landmarks desenhados
        """
        frame = frame.copy()
        
        # Recriar objeto de landmarks do MediaPipe para usar draw_landmarks
        mp_landmarks = self.mp_pose.PoseLandmark
        
        # Criar estrutura de landmarks
        class LandmarkList:
            def __init__(self, landmarks):
                self.landmark = landmarks
        
        class LandmarkPoint:
            def __init__(self, lm: Landmark):
                self.x = lm.x
                self.y = lm.y
                self.z = lm.z
                self.visibility = lm.visibility
        
        landmark_list = LandmarkList([
            LandmarkPoint(lm) for lm in pose_result.landmarks
        ])
        
        self.mp_drawing.draw_landmarks(
            frame,
            landmark_list,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return frame
    
    def get_key_points(self, pose_result: PoseResult) -> Dict[str, Tuple[int, int]]:
        """
        Retorna pontos-chave em coordenadas de pixel.
        
        Args:
            pose_result: Resultado da detecção
        
        Returns:
            Dicionário com nome do ponto e coordenadas (x, y)
        """
        w = pose_result.frame_width
        h = pose_result.frame_height
        
        key_points = {
            'nose': PoseLandmark.NOSE,
            'left_shoulder': PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': PoseLandmark.LEFT_ELBOW,
            'right_elbow': PoseLandmark.RIGHT_ELBOW,
            'left_wrist': PoseLandmark.LEFT_WRIST,
            'right_wrist': PoseLandmark.RIGHT_WRIST,
            'left_hip': PoseLandmark.LEFT_HIP,
            'right_hip': PoseLandmark.RIGHT_HIP,
            'left_knee': PoseLandmark.LEFT_KNEE,
            'right_knee': PoseLandmark.RIGHT_KNEE,
            'left_ankle': PoseLandmark.LEFT_ANKLE,
            'right_ankle': PoseLandmark.RIGHT_ANKLE,
        }
        
        result = {}
        for name, idx in key_points.items():
            landmark = pose_result.get_landmark(idx)
            if landmark and landmark.visibility > 0.5:
                result[name] = landmark.to_pixel(w, h)
        
        return result
    
    def release(self):
        """Libera recursos do detector."""
        self.pose.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# =============================================================================
# Funções utilitárias
# =============================================================================

def calculate_angle(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    point3: Tuple[float, float]
) -> float:
    """
    Calcula o ângulo entre três pontos.
    
    O ângulo é calculado no ponto2 (vértice).
    
    Args:
        point1: Primeiro ponto (x, y)
        point2: Ponto central/vértice (x, y)
        point3: Terceiro ponto (x, y)
    
    Returns:
        Ângulo em graus (0-180)
    """
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    # Vetores
    ba = a - b
    bc = c - b
    
    # Produto escalar e magnitudes
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    
    # Limitar a [-1, 1] para evitar erros numéricos
    cosine = np.clip(cosine, -1.0, 1.0)
    
    # Ângulo em graus
    angle = np.degrees(np.arccos(cosine))
    
    return angle


def calculate_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """
    Calcula a distância euclidiana entre dois pontos.
    
    Args:
        point1: Primeiro ponto (x, y)
        point2: Segundo ponto (x, y)
    
    Returns:
        Distância em pixels
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


if __name__ == "__main__":
    # Teste do módulo
    print("Testando módulo de detecção de pose...")
    
    # Criar detector
    detector = PoseDetector()
    
    # Tentar abrir câmera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera")
        print("Criando imagem de teste...")
        
        # Criar imagem de teste
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            test_frame, 
            "Sem camera - Teste de importacao OK", 
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        result = detector.detect(test_frame)
        print(f"Resultado da detecção (imagem vazia): {result}")
        print("Módulo carregado com sucesso!")
    else:
        print("Câmera aberta com sucesso!")
        print("Pressione 'q' para sair")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar pose
            result = detector.detect(frame)
            
            if result:
                # Desenhar landmarks
                frame = detector.draw_landmarks(frame, result)
                
                # Mostrar informações
                cv2.putText(
                    frame,
                    f"Landmarks detectados: {len(result.landmarks)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Nenhuma pose detectada",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            cv2.imshow("Pose Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    detector.release()
    print("Teste concluído!")
