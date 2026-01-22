"""
=============================================================================
Módulo de Geração de Feedback
=============================================================================
Este módulo gera feedback visual e textual em tempo real para
correção de técnica durante exercícios.

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FeedbackLevel(Enum):
    """Níveis de feedback."""
    GOOD = "good"
    WARNING = "warning"
    BAD = "bad"
    INFO = "info"


@dataclass
class FeedbackMessage:
    """Mensagem de feedback."""
    text: str
    level: FeedbackLevel
    joint: Optional[str] = None


class FeedbackGenerator:
    """
    Gerador de feedback visual e textual.
    
    Fornece feedback em tempo real sobre a execução de exercícios.
    """
    
    # Cores para feedback (BGR)
    COLORS = {
        FeedbackLevel.GOOD: (0, 255, 0),      # Verde
        FeedbackLevel.WARNING: (0, 165, 255),  # Laranja
        FeedbackLevel.BAD: (0, 0, 255),        # Vermelho
        FeedbackLevel.INFO: (255, 255, 0),     # Ciano
    }
    
    # Mensagens de feedback para agachamento
    SQUAT_FEEDBACK = {
        'knee_over_toe': "Não deixe os joelhos passarem dos dedos dos pés",
        'knees_caving': "Mantenha os joelhos alinhados com os pés",
        'back_rounding': "Mantenha as costas retas",
        'insufficient_depth': "Agache mais fundo para melhor ativação muscular",
        'good_depth': "Boa profundidade!",
        'good_form': "Excelente forma!",
        'asymmetric': "Equilibre o peso entre as duas pernas",
    }
    
    # Mensagens de feedback para flexão
    PUSHUP_FEEDBACK = {
        'hips_sagging': "Mantenha o quadril alinhado - não deixe cair",
        'hips_high': "Abaixe o quadril - mantenha o corpo reto",
        'elbows_flared': "Mantenha os cotovelos mais próximos do corpo",
        'insufficient_depth': "Desça mais para melhor amplitude",
        'good_form': "Ótima execução!",
    }
    
    def __init__(self, language: str = "pt-BR"):
        """
        Inicializa o gerador de feedback.
        
        Args:
            language: Idioma do feedback
        """
        self.language = language
        self.current_messages: List[FeedbackMessage] = []
    
    def generate_squat_feedback(
        self, 
        analysis_result: Dict
    ) -> List[FeedbackMessage]:
        """
        Gera feedback para agachamento.
        
        Args:
            analysis_result: Resultado da análise do agachamento
        
        Returns:
            Lista de mensagens de feedback
        """
        messages = []
        
        quality_score = analysis_result.get('quality_score', 0)
        knee_angle = analysis_result.get('knee_angle')
        symmetry = analysis_result.get('symmetry', 1.0)
        body_alignment = analysis_result.get('body_alignment', 0)
        depth = analysis_result.get('depth', 'standing')
        
        # Feedback baseado na profundidade
        if depth == 'deep':
            messages.append(FeedbackMessage(
                text="Boa profundidade! 👍",
                level=FeedbackLevel.GOOD
            ))
        elif depth == 'parallel':
            messages.append(FeedbackMessage(
                text="Profundidade OK - tente ir mais fundo",
                level=FeedbackLevel.INFO
            ))
        elif depth == 'partial' and knee_angle and knee_angle < 160:
            messages.append(FeedbackMessage(
                text="Agache mais fundo para melhor ativação",
                level=FeedbackLevel.WARNING,
                joint='knee'
            ))
        
        # Feedback de simetria
        if symmetry < 0.7:
            messages.append(FeedbackMessage(
                text="Equilibre o peso entre as pernas",
                level=FeedbackLevel.WARNING
            ))
        
        # Feedback de alinhamento
        if body_alignment > 20:
            messages.append(FeedbackMessage(
                text="Mantenha o tronco mais ereto",
                level=FeedbackLevel.BAD
            ))
        elif body_alignment > 10:
            messages.append(FeedbackMessage(
                text="Atenção ao alinhamento do tronco",
                level=FeedbackLevel.WARNING
            ))
        
        # Feedback geral de qualidade
        if quality_score >= 90:
            messages.append(FeedbackMessage(
                text="Excelente execução! ⭐",
                level=FeedbackLevel.GOOD
            ))
        elif quality_score >= 70:
            messages.append(FeedbackMessage(
                text="Boa execução",
                level=FeedbackLevel.GOOD
            ))
        
        self.current_messages = messages
        return messages
    
    def generate_pushup_feedback(
        self, 
        analysis_result: Dict
    ) -> List[FeedbackMessage]:
        """
        Gera feedback para flexão de braço.
        
        Args:
            analysis_result: Resultado da análise da flexão
        
        Returns:
            Lista de mensagens de feedback
        """
        messages = []
        
        quality_score = analysis_result.get('quality_score', 0)
        elbow_angle = analysis_result.get('elbow_angle')
        body_alignment = analysis_result.get('body_alignment', 0)
        phase = analysis_result.get('phase', 'up')
        
        # Feedback de alinhamento
        if body_alignment > 20:
            messages.append(FeedbackMessage(
                text="Mantenha o corpo reto!",
                level=FeedbackLevel.BAD
            ))
        elif body_alignment > 10:
            messages.append(FeedbackMessage(
                text="Atenção à postura do corpo",
                level=FeedbackLevel.WARNING
            ))
        
        # Feedback de amplitude
        if phase == 'down' and elbow_angle and elbow_angle > 110:
            messages.append(FeedbackMessage(
                text="Desça mais para amplitude completa",
                level=FeedbackLevel.WARNING,
                joint='elbow'
            ))
        
        # Feedback geral
        if quality_score >= 90:
            messages.append(FeedbackMessage(
                text="Perfeito! Continue assim! 💪",
                level=FeedbackLevel.GOOD
            ))
        
        self.current_messages = messages
        return messages
    
    def draw_feedback_on_frame(
        self,
        frame: np.ndarray,
        messages: List[FeedbackMessage],
        position: str = "top-right"
    ) -> np.ndarray:
        """
        Desenha feedback no frame de vídeo.
        
        Args:
            frame: Frame de vídeo
            messages: Lista de mensagens
            position: Posição do texto ("top-right", "top-left", "bottom")
        
        Returns:
            Frame com feedback desenhado
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Configurações de texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 30
        padding = 10
        
        # Calcular posição inicial
        if position == "top-right":
            x_base = w - 400
            y_base = 30
        elif position == "top-left":
            x_base = 10
            y_base = 30
        else:  # bottom
            x_base = 10
            y_base = h - (len(messages) * line_height) - padding
        
        # Desenhar fundo semi-transparente
        if messages:
            overlay = frame.copy()
            box_height = len(messages) * line_height + padding * 2
            box_width = 380
            
            if position == "top-right":
                cv2.rectangle(
                    overlay,
                    (w - box_width - padding, y_base - padding),
                    (w - padding, y_base + box_height),
                    (0, 0, 0),
                    -1
                )
            else:
                cv2.rectangle(
                    overlay,
                    (x_base - padding, y_base - padding),
                    (x_base + box_width, y_base + box_height),
                    (0, 0, 0),
                    -1
                )
            
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Desenhar mensagens
        for i, msg in enumerate(messages):
            y = y_base + (i * line_height)
            color = self.COLORS.get(msg.level, (255, 255, 255))
            
            # Ícone de status
            icon = "●"
            if msg.level == FeedbackLevel.GOOD:
                icon = "✓"
            elif msg.level == FeedbackLevel.BAD:
                icon = "✗"
            elif msg.level == FeedbackLevel.WARNING:
                icon = "!"
            
            text = f"{icon} {msg.text}"
            
            if position == "top-right":
                # Alinhar à direita
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                x = w - text_size[0] - padding - 10
            else:
                x = x_base
            
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
        
        return frame
    
    def draw_metrics_overlay(
        self,
        frame: np.ndarray,
        metrics: Dict,
        position: str = "top-left"
    ) -> np.ndarray:
        """
        Desenha overlay com métricas no frame.
        
        Args:
            frame: Frame de vídeo
            metrics: Dicionário de métricas
            position: Posição do overlay
        
        Returns:
            Frame com métricas
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 25
        
        y = 30
        x = 10
        
        # Fundo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (250, 30 + len(metrics) * line_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        for key, value in metrics.items():
            # Formatar valor
            if isinstance(value, float):
                text = f"{key}: {value:.1f}"
            else:
                text = f"{key}: {value}"
            
            cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
            y += line_height
        
        return frame
    
    def draw_angle_visualization(
        self,
        frame: np.ndarray,
        point1: Tuple[int, int],
        vertex: Tuple[int, int],
        point2: Tuple[int, int],
        angle: float,
        color: Tuple[int, int, int] = (0, 255, 255),
        label: str = ""
    ) -> np.ndarray:
        """
        Desenha visualização de ângulo no frame.
        
        Args:
            frame: Frame de vídeo
            point1: Primeiro ponto
            vertex: Vértice (onde o ângulo é medido)
            point2: Segundo ponto
            angle: Ângulo em graus
            color: Cor do desenho
            label: Rótulo opcional
        
        Returns:
            Frame com ângulo visualizado
        """
        frame = frame.copy()
        
        # Desenhar linhas
        cv2.line(frame, vertex, point1, color, 2)
        cv2.line(frame, vertex, point2, color, 2)
        
        # Desenhar arco do ângulo
        # Calcular ângulos para o arco
        angle1 = np.arctan2(point1[1] - vertex[1], point1[0] - vertex[0])
        angle2 = np.arctan2(point2[1] - vertex[1], point2[0] - vertex[0])
        
        start_angle = np.degrees(min(angle1, angle2))
        end_angle = np.degrees(max(angle1, angle2))
        
        # Desenhar arco
        cv2.ellipse(
            frame,
            vertex,
            (30, 30),
            0,
            start_angle,
            end_angle,
            color,
            2
        )
        
        # Desenhar texto com ângulo
        text = f"{angle:.0f}°"
        if label:
            text = f"{label}: {text}"
        
        text_pos = (vertex[0] + 35, vertex[1] - 10)
        cv2.putText(
            frame, 
            text, 
            text_pos, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            2
        )
        
        return frame
    
    def draw_rep_counter(
        self,
        frame: np.ndarray,
        count: int,
        position: str = "top-center"
    ) -> np.ndarray:
        """
        Desenha contador de repetições.
        
        Args:
            frame: Frame de vídeo
            count: Número de repetições
            position: Posição do contador
        
        Returns:
            Frame com contador
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        text = f"Reps: {count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Calcular tamanho do texto
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Posição
        if position == "top-center":
            x = (w - text_size[0]) // 2
            y = 50
        else:
            x = w - text_size[0] - 20
            y = 50
        
        # Fundo
        cv2.rectangle(
            frame,
            (x - 10, y - text_size[1] - 10),
            (x + text_size[0] + 10, y + 10),
            (0, 0, 0),
            -1
        )
        
        # Texto
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 0), thickness)
        
        return frame
    
    def draw_quality_bar(
        self,
        frame: np.ndarray,
        score: float,
        position: Tuple[int, int] = (10, 100)
    ) -> np.ndarray:
        """
        Desenha barra de qualidade do movimento.
        
        Args:
            frame: Frame de vídeo
            score: Score de qualidade (0-100)
            position: Posição (x, y) da barra
        
        Returns:
            Frame com barra de qualidade
        """
        frame = frame.copy()
        x, y = position
        
        bar_width = 200
        bar_height = 20
        
        # Cor baseada no score
        if score >= 80:
            color = (0, 255, 0)  # Verde
        elif score >= 60:
            color = (0, 255, 255)  # Amarelo
        elif score >= 40:
            color = (0, 165, 255)  # Laranja
        else:
            color = (0, 0, 255)  # Vermelho
        
        # Fundo da barra
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
        
        # Barra de progresso
        filled_width = int(bar_width * score / 100)
        cv2.rectangle(frame, (x, y), (x + filled_width, y + bar_height), color, -1)
        
        # Borda
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 1)
        
        # Texto
        text = f"Qualidade: {score:.0f}%"
        cv2.putText(
            frame, 
            text, 
            (x, y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        return frame


if __name__ == "__main__":
    print("Módulo de geração de feedback carregado com sucesso!")
    
    # Teste de visualização
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    generator = FeedbackGenerator()
    
    # Criar mensagens de teste
    messages = [
        FeedbackMessage("Boa profundidade!", FeedbackLevel.GOOD),
        FeedbackMessage("Mantenha o tronco ereto", FeedbackLevel.WARNING),
        FeedbackMessage("Joelhos assimétricos", FeedbackLevel.BAD),
    ]
    
    # Desenhar feedback
    test_frame = generator.draw_feedback_on_frame(test_frame, messages)
    test_frame = generator.draw_rep_counter(test_frame, 5)
    test_frame = generator.draw_quality_bar(test_frame, 75)
    
    # Mostrar
    cv2.imshow("Feedback Test", test_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Teste concluído!")
