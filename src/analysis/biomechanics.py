"""
=============================================================================
Módulo de Análise Biomecânica
=============================================================================
Este módulo implementa cálculos biomecânicos para análise de movimento,
incluindo ângulos articulares, simetria, velocidade e métricas específicas.

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

# Importar do módulo de pose
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pose_detector import PoseResult, PoseLandmark, calculate_angle


class JointType(Enum):
    """Tipos de articulações para análise."""
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


@dataclass
class JointAngle:
    """Representa o ângulo de uma articulação."""
    joint_type: JointType
    angle: float
    confidence: float
    timestamp: float


@dataclass
class BiomechanicsResult:
    """Resultado completo da análise biomecânica."""
    joint_angles: Dict[str, JointAngle]
    symmetry_score: float
    center_of_mass: Tuple[float, float]
    body_alignment: float
    timestamp: float


class BiomechanicsAnalyzer:
    """
    Analisador biomecânico para poses corporais.
    
    Calcula ângulos articulares, simetria, centro de massa
    e outras métricas biomecânicas.
    
    Exemplo de Uso:
        >>> analyzer = BiomechanicsAnalyzer()
        >>> result = analyzer.analyze(pose_result)
        >>> print(f"Ângulo do joelho: {result.joint_angles['left_knee'].angle}°")
    """
    
    # Definição das articulações (ponto1, vértice, ponto2)
    JOINT_DEFINITIONS = {
        JointType.LEFT_ELBOW: (
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.LEFT_ELBOW,
            PoseLandmark.LEFT_WRIST
        ),
        JointType.RIGHT_ELBOW: (
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.RIGHT_WRIST
        ),
        JointType.LEFT_SHOULDER: (
            PoseLandmark.LEFT_ELBOW,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.LEFT_HIP
        ),
        JointType.RIGHT_SHOULDER: (
            PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.RIGHT_HIP
        ),
        JointType.LEFT_HIP: (
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.LEFT_KNEE
        ),
        JointType.RIGHT_HIP: (
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.RIGHT_KNEE
        ),
        JointType.LEFT_KNEE: (
            PoseLandmark.LEFT_HIP,
            PoseLandmark.LEFT_KNEE,
            PoseLandmark.LEFT_ANKLE
        ),
        JointType.RIGHT_KNEE: (
            PoseLandmark.RIGHT_HIP,
            PoseLandmark.RIGHT_KNEE,
            PoseLandmark.RIGHT_ANKLE
        ),
        JointType.LEFT_ANKLE: (
            PoseLandmark.LEFT_KNEE,
            PoseLandmark.LEFT_ANKLE,
            PoseLandmark.LEFT_FOOT_INDEX
        ),
        JointType.RIGHT_ANKLE: (
            PoseLandmark.RIGHT_KNEE,
            PoseLandmark.RIGHT_ANKLE,
            PoseLandmark.RIGHT_FOOT_INDEX
        ),
    }
    
    # Pares simétricos para análise de simetria
    SYMMETRIC_PAIRS = [
        (JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW),
        (JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER),
        (JointType.LEFT_HIP, JointType.RIGHT_HIP),
        (JointType.LEFT_KNEE, JointType.RIGHT_KNEE),
        (JointType.LEFT_ANKLE, JointType.RIGHT_ANKLE),
    ]
    
    def __init__(self, history_size: int = 30):
        """
        Inicializa o analisador biomecânico.
        
        Args:
            history_size: Tamanho do histórico para análise temporal
        """
        self.history_size = history_size
        self.angle_history: Dict[str, deque] = {
            jt.value: deque(maxlen=history_size) 
            for jt in JointType
        }
        self._frame_count = 0
    
    def analyze(self, pose_result: PoseResult) -> BiomechanicsResult:
        """
        Realiza análise biomecânica completa de uma pose.
        
        Args:
            pose_result: Resultado da detecção de pose
        
        Returns:
            BiomechanicsResult com todas as métricas calculadas
        """
        self._frame_count += 1
        timestamp = pose_result.timestamp
        
        # Calcular ângulos de todas as articulações
        joint_angles = self._calculate_all_angles(pose_result, timestamp)
        
        # Calcular simetria
        symmetry_score = self._calculate_symmetry(joint_angles)
        
        # Calcular centro de massa
        center_of_mass = self._calculate_center_of_mass(pose_result)
        
        # Calcular alinhamento corporal
        body_alignment = self._calculate_body_alignment(pose_result)
        
        return BiomechanicsResult(
            joint_angles=joint_angles,
            symmetry_score=symmetry_score,
            center_of_mass=center_of_mass,
            body_alignment=body_alignment,
            timestamp=timestamp
        )
    
    def _calculate_all_angles(
        self, 
        pose_result: PoseResult,
        timestamp: float
    ) -> Dict[str, JointAngle]:
        """Calcula ângulos de todas as articulações."""
        angles = {}
        
        for joint_type, (idx1, idx2, idx3) in self.JOINT_DEFINITIONS.items():
            lm1 = pose_result.get_landmark(idx1)
            lm2 = pose_result.get_landmark(idx2)
            lm3 = pose_result.get_landmark(idx3)
            
            if all([lm1, lm2, lm3]):
                # Calcular confiança média
                confidence = (lm1.visibility + lm2.visibility + lm3.visibility) / 3
                
                if confidence > 0.5:
                    # Calcular ângulo
                    angle = calculate_angle(
                        (lm1.x, lm1.y),
                        (lm2.x, lm2.y),
                        (lm3.x, lm3.y)
                    )
                    
                    joint_angle = JointAngle(
                        joint_type=joint_type,
                        angle=angle,
                        confidence=confidence,
                        timestamp=timestamp
                    )
                    
                    angles[joint_type.value] = joint_angle
                    
                    # Adicionar ao histórico
                    self.angle_history[joint_type.value].append(angle)
        
        return angles
    
    def _calculate_symmetry(
        self, 
        joint_angles: Dict[str, JointAngle]
    ) -> float:
        """
        Calcula score de simetria bilateral.
        
        Retorna valor entre 0 (assimétrico) e 1 (perfeitamente simétrico).
        """
        differences = []
        
        for left_joint, right_joint in self.SYMMETRIC_PAIRS:
            left = joint_angles.get(left_joint.value)
            right = joint_angles.get(right_joint.value)
            
            if left and right:
                # Diferença normalizada (máximo 180 graus)
                diff = abs(left.angle - right.angle) / 180.0
                differences.append(diff)
        
        if not differences:
            return 0.0
        
        # Score: 1 - média das diferenças normalizadas
        avg_diff = np.mean(differences)
        symmetry = 1.0 - min(avg_diff * 5, 1.0)  # Amplificar diferenças pequenas
        
        return max(0.0, symmetry)
    
    def _calculate_center_of_mass(
        self, 
        pose_result: PoseResult
    ) -> Tuple[float, float]:
        """
        Calcula o centro de massa aproximado do corpo.
        
        Usa pesos aproximados para diferentes partes do corpo.
        """
        # Pesos aproximados das partes do corpo (% massa corporal)
        segment_weights = {
            PoseLandmark.LEFT_SHOULDER: 0.08,
            PoseLandmark.RIGHT_SHOULDER: 0.08,
            PoseLandmark.LEFT_HIP: 0.15,
            PoseLandmark.RIGHT_HIP: 0.15,
            PoseLandmark.LEFT_KNEE: 0.05,
            PoseLandmark.RIGHT_KNEE: 0.05,
            PoseLandmark.LEFT_ANKLE: 0.02,
            PoseLandmark.RIGHT_ANKLE: 0.02,
        }
        
        total_weight = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        
        for landmark_idx, weight in segment_weights.items():
            lm = pose_result.get_landmark(landmark_idx)
            if lm and lm.visibility > 0.5:
                weighted_x += lm.x * weight
                weighted_y += lm.y * weight
                total_weight += weight
        
        if total_weight > 0:
            return (weighted_x / total_weight, weighted_y / total_weight)
        
        return (0.5, 0.5)  # Centro da imagem como fallback
    
    def _calculate_body_alignment(self, pose_result: PoseResult) -> float:
        """
        Calcula o alinhamento vertical do corpo.
        
        Verifica se ombros e quadris estão alinhados verticalmente.
        Retorna ângulo de desvio em graus.
        """
        left_shoulder = pose_result.get_landmark(PoseLandmark.LEFT_SHOULDER)
        right_shoulder = pose_result.get_landmark(PoseLandmark.RIGHT_SHOULDER)
        left_hip = pose_result.get_landmark(PoseLandmark.LEFT_HIP)
        right_hip = pose_result.get_landmark(PoseLandmark.RIGHT_HIP)
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0.0
        
        # Centro dos ombros
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Centro do quadril
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # Calcular ângulo de inclinação do tronco
        dx = hip_center_x - shoulder_center_x
        dy = hip_center_y - shoulder_center_y
        
        # Ângulo em relação à vertical (0° = perfeitamente vertical)
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        
        return angle
    
    def get_angle(self, joint_type: JointType) -> Optional[float]:
        """
        Retorna o ângulo atual de uma articulação.
        
        Args:
            joint_type: Tipo da articulação
        
        Returns:
            Ângulo em graus ou None se não disponível
        """
        history = self.angle_history.get(joint_type.value)
        if history and len(history) > 0:
            return history[-1]
        return None
    
    def get_angle_velocity(self, joint_type: JointType) -> Optional[float]:
        """
        Calcula a velocidade angular de uma articulação.
        
        Args:
            joint_type: Tipo da articulação
        
        Returns:
            Velocidade angular em graus/frame ou None
        """
        history = self.angle_history.get(joint_type.value)
        if history and len(history) >= 2:
            return history[-1] - history[-2]
        return None
    
    def get_smoothed_angle(
        self, 
        joint_type: JointType, 
        window: int = 5
    ) -> Optional[float]:
        """
        Retorna ângulo suavizado usando média móvel.
        
        Args:
            joint_type: Tipo da articulação
            window: Tamanho da janela de suavização
        
        Returns:
            Ângulo suavizado ou None
        """
        history = self.angle_history.get(joint_type.value)
        if history and len(history) >= window:
            return np.mean(list(history)[-window:])
        elif history and len(history) > 0:
            return np.mean(list(history))
        return None


class ExerciseAnalyzer:
    """
    Analisador específico para exercícios.
    
    Fornece análise detalhada para tipos específicos de exercícios.
    """
    
    def __init__(self):
        self.biomechanics = BiomechanicsAnalyzer()
        self.rep_count = 0
        self.current_phase = "neutral"
        self._last_knee_angle = None
    
    def analyze_squat(
        self, 
        pose_result: PoseResult
    ) -> Dict[str, any]:
        """
        Analisa execução de agachamento.
        
        Args:
            pose_result: Resultado da detecção de pose
        
        Returns:
            Dicionário com métricas do agachamento
        """
        bio_result = self.biomechanics.analyze(pose_result)
        
        # Obter ângulos dos joelhos
        left_knee = bio_result.joint_angles.get('left_knee')
        right_knee = bio_result.joint_angles.get('right_knee')
        
        # Obter ângulos do quadril
        left_hip = bio_result.joint_angles.get('left_hip')
        right_hip = bio_result.joint_angles.get('right_hip')
        
        # Calcular métricas
        knee_angle = None
        hip_angle = None
        
        if left_knee and right_knee:
            knee_angle = (left_knee.angle + right_knee.angle) / 2
        
        if left_hip and right_hip:
            hip_angle = (left_hip.angle + right_hip.angle) / 2
        
        # Determinar profundidade do agachamento
        depth = "standing"
        if knee_angle:
            if knee_angle < 100:
                depth = "deep"
            elif knee_angle < 130:
                depth = "parallel"
            elif knee_angle < 160:
                depth = "partial"
        
        # Contar repetições
        if knee_angle and self._last_knee_angle:
            # Detectar transição para agachamento profundo
            if self._last_knee_angle > 130 and knee_angle < 100:
                if self.current_phase != "down":
                    self.current_phase = "down"
            # Detectar retorno à posição em pé
            elif self._last_knee_angle < 130 and knee_angle > 160:
                if self.current_phase == "down":
                    self.rep_count += 1
                    self.current_phase = "up"
        
        self._last_knee_angle = knee_angle
        
        # Avaliar qualidade
        feedback = []
        quality_score = 100
        
        if knee_angle:
            # Verificar simetria dos joelhos
            if left_knee and right_knee:
                knee_diff = abs(left_knee.angle - right_knee.angle)
                if knee_diff > 10:
                    feedback.append("Joelhos assimétricos - equilibre o peso")
                    quality_score -= 15
        
        # Verificar alinhamento do tronco
        if bio_result.body_alignment > 15:
            feedback.append("Mantenha o tronco mais ereto")
            quality_score -= 20
        
        # Verificar simetria geral
        if bio_result.symmetry_score < 0.7:
            feedback.append("Melhore a simetria do movimento")
            quality_score -= 10
        
        return {
            'knee_angle': knee_angle,
            'hip_angle': hip_angle,
            'depth': depth,
            'rep_count': self.rep_count,
            'phase': self.current_phase,
            'symmetry': bio_result.symmetry_score,
            'body_alignment': bio_result.body_alignment,
            'quality_score': max(0, quality_score),
            'feedback': feedback,
            'biomechanics': bio_result
        }
    
    def analyze_pushup(
        self, 
        pose_result: PoseResult
    ) -> Dict[str, any]:
        """
        Analisa execução de flexão de braço.
        
        Args:
            pose_result: Resultado da detecção de pose
        
        Returns:
            Dicionário com métricas da flexão
        """
        bio_result = self.biomechanics.analyze(pose_result)
        
        # Obter ângulos dos cotovelos
        left_elbow = bio_result.joint_angles.get('left_elbow')
        right_elbow = bio_result.joint_angles.get('right_elbow')
        
        elbow_angle = None
        if left_elbow and right_elbow:
            elbow_angle = (left_elbow.angle + right_elbow.angle) / 2
        
        # Determinar fase
        phase = "up"
        if elbow_angle:
            if elbow_angle < 100:
                phase = "down"
        
        # Avaliar qualidade
        feedback = []
        quality_score = 100
        
        # Verificar alinhamento do corpo
        if bio_result.body_alignment > 15:
            feedback.append("Mantenha o corpo reto - não deixe o quadril cair")
            quality_score -= 25
        
        # Verificar simetria dos cotovelos
        if left_elbow and right_elbow:
            elbow_diff = abs(left_elbow.angle - right_elbow.angle)
            if elbow_diff > 15:
                feedback.append("Braços assimétricos - equilibre a força")
                quality_score -= 15
        
        return {
            'elbow_angle': elbow_angle,
            'phase': phase,
            'body_alignment': bio_result.body_alignment,
            'symmetry': bio_result.symmetry_score,
            'quality_score': max(0, quality_score),
            'feedback': feedback,
            'biomechanics': bio_result
        }
    
    def reset(self):
        """Reseta contadores e estado."""
        self.rep_count = 0
        self.current_phase = "neutral"
        self._last_knee_angle = None


if __name__ == "__main__":
    print("Módulo de análise biomecânica carregado com sucesso!")
    print("\nArticulações disponíveis:")
    for jt in JointType:
        print(f"  - {jt.value}")
    
    print("\nPares simétricos para análise:")
    for left, right in BiomechanicsAnalyzer.SYMMETRIC_PAIRS:
        print(f"  - {left.value} <-> {right.value}")
