"""
=============================================================================
Testes do Pipeline de Análise de Movimento
=============================================================================
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPoseDetector:
    """Testes para o módulo de detecção de pose."""
    
    def test_import(self):
        """Testa se o módulo pode ser importado."""
        from preprocessing.pose_detector import PoseDetector, PoseLandmark
        assert PoseDetector is not None
        assert PoseLandmark is not None
    
    def test_pose_detector_creation(self):
        """Testa criação do detector."""
        from preprocessing.pose_detector import PoseDetector
        detector = PoseDetector()
        assert detector is not None
        detector.release()
    
    def test_calculate_angle(self):
        """Testa cálculo de ângulo entre pontos."""
        from preprocessing.pose_detector import calculate_angle
        
        # Ângulo de 90 graus
        angle = calculate_angle((0, 0), (0, 1), (1, 1))
        assert abs(angle - 90) < 1  # Tolerância de 1 grau
        
        # Ângulo de 180 graus (linha reta)
        angle = calculate_angle((0, 0), (1, 0), (2, 0))
        assert abs(angle - 180) < 1
        
        # Ângulo de 45 graus
        angle = calculate_angle((0, 1), (0, 0), (1, 0))
        assert abs(angle - 90) < 1


class TestBiomechanics:
    """Testes para o módulo de análise biomecânica."""
    
    def test_import(self):
        """Testa se o módulo pode ser importado."""
        from analysis.biomechanics import BiomechanicsAnalyzer, JointType
        assert BiomechanicsAnalyzer is not None
        assert JointType is not None
    
    def test_joint_types(self):
        """Testa tipos de articulações."""
        from analysis.biomechanics import JointType
        
        assert JointType.LEFT_KNEE.value == "left_knee"
        assert JointType.RIGHT_KNEE.value == "right_knee"
        assert JointType.LEFT_ELBOW.value == "left_elbow"
    
    def test_exercise_analyzer_creation(self):
        """Testa criação do analisador de exercícios."""
        from analysis.biomechanics import ExerciseAnalyzer
        
        analyzer = ExerciseAnalyzer()
        assert analyzer is not None
        assert analyzer.rep_count == 0


class TestFeedbackGenerator:
    """Testes para o módulo de geração de feedback."""
    
    def test_import(self):
        """Testa se o módulo pode ser importado."""
        from analysis.feedback_generator import FeedbackGenerator, FeedbackLevel
        assert FeedbackGenerator is not None
        assert FeedbackLevel is not None
    
    def test_feedback_levels(self):
        """Testa níveis de feedback."""
        from analysis.feedback_generator import FeedbackLevel
        
        assert FeedbackLevel.GOOD.value == "good"
        assert FeedbackLevel.WARNING.value == "warning"
        assert FeedbackLevel.BAD.value == "bad"
    
    def test_feedback_generator_creation(self):
        """Testa criação do gerador de feedback."""
        from analysis.feedback_generator import FeedbackGenerator
        
        generator = FeedbackGenerator()
        assert generator is not None


class TestRealtimeAnalyzer:
    """Testes para o módulo de análise em tempo real."""
    
    def test_import(self):
        """Testa se o módulo pode ser importado."""
        from inference.realtime_analyzer import SportsMotionAnalyzer, ExerciseType
        assert SportsMotionAnalyzer is not None
        assert ExerciseType is not None
    
    def test_exercise_types(self):
        """Testa tipos de exercícios."""
        from inference.realtime_analyzer import ExerciseType
        
        assert ExerciseType.SQUAT.value == "squat"
        assert ExerciseType.PUSHUP.value == "pushup"
        assert ExerciseType.GENERAL.value == "general"


class TestIntegration:
    """Testes de integração."""
    
    def test_full_pipeline_creation(self):
        """Testa criação completa do pipeline."""
        from preprocessing.pose_detector import PoseDetector
        from analysis.biomechanics import BiomechanicsAnalyzer, ExerciseAnalyzer
        from analysis.feedback_generator import FeedbackGenerator
        
        # Criar todos os componentes
        detector = PoseDetector()
        biomechanics = BiomechanicsAnalyzer()
        exercise = ExerciseAnalyzer()
        feedback = FeedbackGenerator()
        
        assert all([detector, biomechanics, exercise, feedback])
        
        # Limpar
        detector.release()
    
    def test_fake_frame_processing(self):
        """Testa processamento de frame falso."""
        from preprocessing.pose_detector import PoseDetector
        import numpy as np
        
        detector = PoseDetector()
        
        # Criar frame preto (nenhuma pose deve ser detectada)
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(fake_frame)
        
        # Sem pose em frame preto
        assert result is None
        
        detector.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
