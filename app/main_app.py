"""
=============================================================================
Aplicação Principal - Analisador de Movimento Esportivo
=============================================================================
Este módulo implementa a aplicação principal para análise de movimento
em tempo real usando linha de comando e OpenCV.

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import cv2
import numpy as np
import sys
import os
import time
from collections import deque
from typing import Optional

# Adicionar path do src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from preprocessing.pose_detector import PoseDetector, PoseLandmark
    from analysis.biomechanics import BiomechanicsAnalyzer, ExerciseAnalyzer
    from analysis.feedback_generator import FeedbackGenerator, FeedbackLevel
    from inference.realtime_analyzer import SportsMotionAnalyzer, ExerciseType
except ImportError as e:
    print(f"Erro de importação: {e}")
    print("Certifique-se de que todas as dependências estão instaladas.")
    print("Execute: pip install -r requirements.txt")
    sys.exit(1)


class CLIApp:
    """
    Aplicação de linha de comando para análise de movimento.
    
    Interface simples usando apenas OpenCV para visualização.
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        video_path: Optional[str] = None,
        exercise_type: str = 'squat'
    ):
        """
        Inicializa a aplicação.
        
        Args:
            camera_index: Índice da câmera
            video_path: Caminho para vídeo (opcional)
            exercise_type: Tipo de exercício
        """
        self.camera_index = camera_index
        self.video_path = video_path
        self.exercise_type = exercise_type
        
        # Inicializar analisador
        self.analyzer = SportsMotionAnalyzer(
            exercise_type=exercise_type,
            camera_index=camera_index,
            video_path=video_path
        )
        
        print("="*60)
        print("   ANALISADOR DE MOVIMENTO ESPORTIVO")
        print("="*60)
        print(f"   Exercício: {exercise_type.upper()}")
        print("="*60)
        print("\n   CONTROLES:")
        print("   [Q] - Sair")
        print("   [R] - Resetar contador")
        print("   [S] - Screenshot")
        print("   [1] - Agachamento")
        print("   [2] - Flexão")
        print("   [3] - Modo geral")
        print("="*60)
    
    def run(self):
        """Executa a aplicação."""
        self.analyzer.start_realtime_analysis()


def print_banner():
    """Imprime banner de início."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🏃 ANALISADOR DE MOVIMENTO ESPORTIVO 🏋️               ║
    ║                                                           ║
    ║     Sistema de análise biomecânica em tempo real          ║
    ║     usando visão computacional e MediaPipe                ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Função principal."""
    import argparse
    
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Analisador de Movimento Esportivo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python -m app.main_app                    # Usar webcam, modo agachamento
  python -m app.main_app -e pushup          # Modo flexão
  python -m app.main_app -v video.mp4       # Analisar vídeo
  python -m app.main_app -c 1               # Usar câmera índice 1
        """
    )
    
    parser.add_argument(
        '--exercise', '-e',
        type=str,
        default='squat',
        choices=['squat', 'pushup', 'general'],
        help='Tipo de exercício para análise (default: squat)'
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Índice da câmera (default: 0)'
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        default=None,
        help='Caminho para arquivo de vídeo (opcional)'
    )
    
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Forçar modo CLI (linha de comando)'
    )
    
    args = parser.parse_args()
    
    # Verificar fonte de vídeo
    if args.video:
        if not os.path.exists(args.video):
            print(f"Erro: Arquivo de vídeo não encontrado: {args.video}")
            sys.exit(1)
        print(f"Usando vídeo: {args.video}")
    else:
        print(f"Usando câmera: {args.camera}")
    
    print(f"Exercício: {args.exercise}")
    print()
    
    # Executar aplicação
    try:
        app = CLIApp(
            camera_index=args.camera,
            video_path=args.video,
            exercise_type=args.exercise
        )
        app.run()
    except KeyboardInterrupt:
        print("\nAplicação encerrada pelo usuário.")
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
