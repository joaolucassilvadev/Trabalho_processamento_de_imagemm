"""
=============================================================================
Coletor de Dataset para Treinamento
=============================================================================
Este módulo coleta dados de landmarks do MediaPipe para criar datasets
de treinamento para classificação de qualidade de movimento.

Uso:
    python -m src.data.dataset_collector --exercise squat --output data/squat_dataset.csv

Autor: Projeto Acadêmico
Data: 2024
=============================================================================
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import time
from typing import List, Dict, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.pose_detector import PoseDetector, PoseResult, PoseLandmark


class DatasetCollector:
    """
    Coletor de dados de pose para criar datasets de treinamento.

    Captura landmarks do MediaPipe e permite rotular como:
    - 0: Movimento incorreto/ruim
    - 1: Movimento parcialmente correto
    - 2: Movimento correto/bom
    """

    # Landmarks importantes para análise de exercícios
    IMPORTANT_LANDMARKS = [
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
        PoseLandmark.RIGHT_ANKLE,
    ]

    def __init__(self, exercise_type: str = 'squat'):
        """
        Inicializa o coletor.

        Args:
            exercise_type: Tipo de exercício ('squat', 'pushup', 'running', etc.)
        """
        self.exercise_type = exercise_type
        self.pose_detector = PoseDetector(model_complexity=1)
        self.data: List[Dict] = []
        self.current_label = 2  # Padrão: bom

    def extract_features(self, pose_result: PoseResult) -> Optional[Dict]:
        """
        Extrai features dos landmarks para o dataset.

        Args:
            pose_result: Resultado da detecção de pose

        Returns:
            Dicionário com features ou None se landmarks insuficientes
        """
        features = {
            'timestamp': time.time(),
            'exercise': self.exercise_type,
        }

        # Extrair coordenadas x, y, z e visibilidade de cada landmark importante
        for lm_idx in self.IMPORTANT_LANDMARKS:
            lm = pose_result.get_landmark(lm_idx)
            lm_name = PoseLandmark(lm_idx).name.lower()

            if lm and lm.visibility > 0.5:
                features[f'{lm_name}_x'] = lm.x
                features[f'{lm_name}_y'] = lm.y
                features[f'{lm_name}_z'] = lm.z
                features[f'{lm_name}_vis'] = lm.visibility
            else:
                # Landmark não visível
                features[f'{lm_name}_x'] = 0.0
                features[f'{lm_name}_y'] = 0.0
                features[f'{lm_name}_z'] = 0.0
                features[f'{lm_name}_vis'] = 0.0

        # Calcular ângulos articulares como features adicionais
        angles = self._calculate_angles(pose_result)
        features.update(angles)

        return features

    def _calculate_angles(self, pose_result: PoseResult) -> Dict[str, float]:
        """Calcula ângulos articulares importantes."""
        angles = {}

        def get_point(idx):
            lm = pose_result.get_landmark(idx)
            if lm and lm.visibility > 0.5:
                return (lm.x, lm.y)
            return None

        def calc_angle(p1, p2, p3):
            if not all([p1, p2, p3]):
                return 0.0
            a = np.array(p1)
            b = np.array(p2)
            c = np.array(p3)
            ba = a - b
            bc = c - b
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        # Ângulos dos joelhos
        angles['left_knee_angle'] = calc_angle(
            get_point(PoseLandmark.LEFT_HIP),
            get_point(PoseLandmark.LEFT_KNEE),
            get_point(PoseLandmark.LEFT_ANKLE)
        )
        angles['right_knee_angle'] = calc_angle(
            get_point(PoseLandmark.RIGHT_HIP),
            get_point(PoseLandmark.RIGHT_KNEE),
            get_point(PoseLandmark.RIGHT_ANKLE)
        )

        # Ângulos do quadril
        angles['left_hip_angle'] = calc_angle(
            get_point(PoseLandmark.LEFT_SHOULDER),
            get_point(PoseLandmark.LEFT_HIP),
            get_point(PoseLandmark.LEFT_KNEE)
        )
        angles['right_hip_angle'] = calc_angle(
            get_point(PoseLandmark.RIGHT_SHOULDER),
            get_point(PoseLandmark.RIGHT_HIP),
            get_point(PoseLandmark.RIGHT_KNEE)
        )

        # Ângulos dos cotovelos
        angles['left_elbow_angle'] = calc_angle(
            get_point(PoseLandmark.LEFT_SHOULDER),
            get_point(PoseLandmark.LEFT_ELBOW),
            get_point(PoseLandmark.LEFT_WRIST)
        )
        angles['right_elbow_angle'] = calc_angle(
            get_point(PoseLandmark.RIGHT_SHOULDER),
            get_point(PoseLandmark.RIGHT_ELBOW),
            get_point(PoseLandmark.RIGHT_WRIST)
        )

        # Ângulos dos ombros
        angles['left_shoulder_angle'] = calc_angle(
            get_point(PoseLandmark.LEFT_ELBOW),
            get_point(PoseLandmark.LEFT_SHOULDER),
            get_point(PoseLandmark.LEFT_HIP)
        )
        angles['right_shoulder_angle'] = calc_angle(
            get_point(PoseLandmark.RIGHT_ELBOW),
            get_point(PoseLandmark.RIGHT_SHOULDER),
            get_point(PoseLandmark.RIGHT_HIP)
        )

        return angles

    def collect_from_camera(
        self,
        camera_index: int = 0,
        output_file: str = 'dataset.csv',
        auto_label: bool = False
    ):
        """
        Coleta dados da câmera em tempo real.

        Controles:
            - 0, 1, 2: Define label (ruim, médio, bom)
            - SPACE: Captura frame com label atual
            - A: Ativa/desativa captura automática
            - S: Salva dataset
            - Q: Sair

        Args:
            camera_index: Índice da câmera
            output_file: Arquivo de saída
            auto_label: Se True, captura automaticamente
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a câmera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        auto_capture = auto_label
        frame_count = 0
        capture_interval = 5  # Capturar a cada N frames

        print("="*60)
        print("COLETOR DE DATASET")
        print("="*60)
        print(f"Exercício: {self.exercise_type}")
        print("\nControles:")
        print("  0/1/2 - Definir label (ruim/médio/bom)")
        print("  SPACE - Capturar frame")
        print("  A - Toggle captura automática")
        print("  S - Salvar dataset")
        print("  Q - Sair")
        print("="*60)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detectar pose
            pose_result = self.pose_detector.detect(frame)

            if pose_result:
                # Desenhar landmarks
                frame = self.pose_detector.draw_landmarks(frame, pose_result)

                # Captura automática
                if auto_capture and frame_count % capture_interval == 0:
                    features = self.extract_features(pose_result)
                    if features:
                        features['label'] = self.current_label
                        self.data.append(features)

            # Desenhar informações
            self._draw_info(frame, auto_capture)

            cv2.imshow('Dataset Collector', frame)

            # Processar teclas
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('0'):
                self.current_label = 0
                print("Label: 0 (Ruim)")
            elif key == ord('1'):
                self.current_label = 1
                print("Label: 1 (Médio)")
            elif key == ord('2'):
                self.current_label = 2
                print("Label: 2 (Bom)")
            elif key == ord(' '):
                if pose_result:
                    features = self.extract_features(pose_result)
                    if features:
                        features['label'] = self.current_label
                        self.data.append(features)
                        print(f"Capturado! Total: {len(self.data)}")
            elif key == ord('a'):
                auto_capture = not auto_capture
                print(f"Captura automática: {'ON' if auto_capture else 'OFF'}")
            elif key == ord('s'):
                self.save_dataset(output_file)

        cap.release()
        cv2.destroyAllWindows()
        self.pose_detector.release()

        # Salvar ao sair
        if self.data:
            self.save_dataset(output_file)

    def collect_from_video(
        self,
        video_path: str,
        label: int,
        output_file: str = 'dataset.csv'
    ):
        """
        Coleta dados de um arquivo de vídeo.

        Args:
            video_path: Caminho do vídeo
            label: Label para todos os frames (0, 1, ou 2)
            output_file: Arquivo de saída
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erro: Não foi possível abrir {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Processando {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps:.1f}")

        frame_count = 0
        captured = 0
        sample_rate = max(1, int(fps / 10))  # ~10 amostras por segundo

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Amostrar a cada N frames
            if frame_count % sample_rate != 0:
                continue

            pose_result = self.pose_detector.detect(frame)

            if pose_result:
                features = self.extract_features(pose_result)
                if features:
                    features['label'] = label
                    features['source'] = video_path
                    self.data.append(features)
                    captured += 1

            # Progresso
            if frame_count % 100 == 0:
                progress = frame_count / total_frames * 100
                print(f"Progresso: {progress:.1f}% ({captured} amostras)")

        cap.release()
        self.pose_detector.release()

        print(f"Concluído! {captured} amostras coletadas")
        self.save_dataset(output_file)

    def _draw_info(self, frame: np.ndarray, auto_capture: bool):
        """Desenha informações no frame."""
        h, w = frame.shape[:2]

        # Fundo
        cv2.rectangle(frame, (10, 10), (350, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 130), (255, 255, 255), 1)

        # Textos
        label_names = {0: 'RUIM', 1: 'MEDIO', 2: 'BOM'}
        label_colors = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 255, 0)}

        cv2.putText(frame, f"Exercicio: {self.exercise_type.upper()}",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Label: {self.current_label} ({label_names[self.current_label]})",
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_colors[self.current_label], 2)
        cv2.putText(frame, f"Amostras: {len(self.data)}",
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Auto: {'ON' if auto_capture else 'OFF'}",
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 255, 0) if auto_capture else (128, 128, 128), 2)

    def save_dataset(self, output_file: str):
        """Salva dataset em CSV."""
        if not self.data:
            print("Nenhum dado para salvar")
            return

        df = pd.DataFrame(self.data)

        # Criar diretório se necessário
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

        # Append se arquivo existir
        if os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(output_file, index=False)
        print(f"Dataset salvo: {output_file} ({len(df)} amostras total)")

    def load_dataset(self, input_file: str) -> pd.DataFrame:
        """Carrega dataset de CSV."""
        return pd.read_csv(input_file)


def generate_synthetic_dataset(
    exercise_type: str,
    n_samples: int = 1000,
    output_file: str = 'synthetic_dataset.csv'
):
    """
    Gera dataset sintético para treinamento inicial.

    Cria variações de poses com labels baseados em regras biomecânicas.

    Args:
        exercise_type: Tipo de exercício
        n_samples: Número de amostras
        output_file: Arquivo de saída
    """
    print(f"Gerando {n_samples} amostras sintéticas para {exercise_type}...")

    data = []

    for i in range(n_samples):
        sample = {'exercise': exercise_type}

        # Gerar pose base
        if exercise_type == 'squat':
            sample = _generate_squat_sample()
        elif exercise_type == 'pushup':
            sample = _generate_pushup_sample()
        else:
            sample = _generate_general_sample()

        sample['exercise'] = exercise_type
        data.append(sample)

        if (i + 1) % 100 == 0:
            print(f"Gerado: {i + 1}/{n_samples}")

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else 'data', exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Dataset sintético salvo: {output_file}")

    return df


def _generate_squat_sample() -> Dict:
    """Gera uma amostra sintética de agachamento."""
    sample = {}

    # Determinar qualidade aleatória
    quality = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])

    # Base positions (normalized 0-1)
    # Em pé: joelhos ~170°, quadril ~170°
    # Agachamento bom: joelhos ~90°, quadril ~90°

    if quality == 2:  # Bom
        knee_angle = np.random.uniform(80, 100)
        hip_angle = np.random.uniform(80, 100)
        symmetry_error = np.random.uniform(0, 5)
    elif quality == 1:  # Médio
        knee_angle = np.random.uniform(100, 140)
        hip_angle = np.random.uniform(100, 140)
        symmetry_error = np.random.uniform(5, 15)
    else:  # Ruim
        knee_angle = np.random.uniform(140, 175)
        hip_angle = np.random.uniform(140, 175)
        symmetry_error = np.random.uniform(15, 30)

    # Adicionar ruído
    noise = np.random.uniform(-5, 5)

    sample['left_knee_angle'] = knee_angle + noise
    sample['right_knee_angle'] = knee_angle + noise + np.random.uniform(-symmetry_error, symmetry_error)
    sample['left_hip_angle'] = hip_angle + noise
    sample['right_hip_angle'] = hip_angle + noise + np.random.uniform(-symmetry_error, symmetry_error)

    # Cotovelos e ombros (menos relevantes para squat)
    sample['left_elbow_angle'] = np.random.uniform(150, 180)
    sample['right_elbow_angle'] = np.random.uniform(150, 180)
    sample['left_shoulder_angle'] = np.random.uniform(20, 60)
    sample['right_shoulder_angle'] = np.random.uniform(20, 60)

    # Gerar coordenadas simuladas dos landmarks
    _add_landmark_coords(sample, quality)

    sample['label'] = quality
    sample['timestamp'] = time.time()

    return sample


def _generate_pushup_sample() -> Dict:
    """Gera uma amostra sintética de flexão."""
    sample = {}

    quality = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])

    if quality == 2:  # Bom
        elbow_angle = np.random.uniform(80, 100)
        body_alignment = np.random.uniform(0, 10)
        symmetry_error = np.random.uniform(0, 5)
    elif quality == 1:  # Médio
        elbow_angle = np.random.uniform(100, 140)
        body_alignment = np.random.uniform(10, 20)
        symmetry_error = np.random.uniform(5, 15)
    else:  # Ruim
        elbow_angle = np.random.uniform(140, 175)
        body_alignment = np.random.uniform(20, 40)
        symmetry_error = np.random.uniform(15, 30)

    noise = np.random.uniform(-5, 5)

    sample['left_elbow_angle'] = elbow_angle + noise
    sample['right_elbow_angle'] = elbow_angle + noise + np.random.uniform(-symmetry_error, symmetry_error)
    sample['left_shoulder_angle'] = np.random.uniform(60, 90) + noise
    sample['right_shoulder_angle'] = np.random.uniform(60, 90) + noise

    # Joelhos e quadril (mais retos em flexão)
    sample['left_knee_angle'] = np.random.uniform(160, 180)
    sample['right_knee_angle'] = np.random.uniform(160, 180)
    sample['left_hip_angle'] = 180 - body_alignment + np.random.uniform(-5, 5)
    sample['right_hip_angle'] = 180 - body_alignment + np.random.uniform(-5, 5)

    _add_landmark_coords(sample, quality)

    sample['label'] = quality
    sample['timestamp'] = time.time()

    return sample


def _generate_general_sample() -> Dict:
    """Gera uma amostra sintética genérica."""
    sample = {}

    quality = np.random.choice([0, 1, 2], p=[0.33, 0.33, 0.34])

    # Ângulos aleatórios
    for angle_name in ['left_knee_angle', 'right_knee_angle', 'left_hip_angle',
                       'right_hip_angle', 'left_elbow_angle', 'right_elbow_angle',
                       'left_shoulder_angle', 'right_shoulder_angle']:
        sample[angle_name] = np.random.uniform(30, 180)

    _add_landmark_coords(sample, quality)

    sample['label'] = quality
    sample['timestamp'] = time.time()

    return sample


def _add_landmark_coords(sample: Dict, quality: int):
    """Adiciona coordenadas simuladas de landmarks."""
    landmarks = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    # Coordenadas base (posição de pé centralizada)
    base_coords = {
        'nose': (0.5, 0.15),
        'left_shoulder': (0.4, 0.25),
        'right_shoulder': (0.6, 0.25),
        'left_elbow': (0.35, 0.4),
        'right_elbow': (0.65, 0.4),
        'left_wrist': (0.3, 0.55),
        'right_wrist': (0.7, 0.55),
        'left_hip': (0.45, 0.5),
        'right_hip': (0.55, 0.5),
        'left_knee': (0.45, 0.7),
        'right_knee': (0.55, 0.7),
        'left_ankle': (0.45, 0.9),
        'right_ankle': (0.55, 0.9),
    }

    # Adicionar ruído baseado na qualidade
    noise_level = {0: 0.1, 1: 0.05, 2: 0.02}[quality]

    for lm in landmarks:
        base_x, base_y = base_coords[lm]
        sample[f'{lm}_x'] = base_x + np.random.uniform(-noise_level, noise_level)
        sample[f'{lm}_y'] = base_y + np.random.uniform(-noise_level, noise_level)
        sample[f'{lm}_z'] = np.random.uniform(-0.5, 0.5)
        sample[f'{lm}_vis'] = np.random.uniform(0.8, 1.0) if quality > 0 else np.random.uniform(0.5, 0.9)


def main():
    """Função principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Coletor de Dataset')
    parser.add_argument('--exercise', '-e', type=str, default='squat',
                       choices=['squat', 'pushup', 'running', 'general'],
                       help='Tipo de exercício')
    parser.add_argument('--output', '-o', type=str, default='data/dataset.csv',
                       help='Arquivo de saída')
    parser.add_argument('--video', '-v', type=str, default=None,
                       help='Processar vídeo em vez de câmera')
    parser.add_argument('--label', '-l', type=int, default=2,
                       help='Label para vídeo (0=ruim, 1=médio, 2=bom)')
    parser.add_argument('--synthetic', '-s', action='store_true',
                       help='Gerar dataset sintético')
    parser.add_argument('--samples', '-n', type=int, default=1000,
                       help='Número de amostras sintéticas')

    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_dataset(args.exercise, args.samples, args.output)
    elif args.video:
        collector = DatasetCollector(args.exercise)
        collector.collect_from_video(args.video, args.label, args.output)
    else:
        collector = DatasetCollector(args.exercise)
        collector.collect_from_camera(output_file=args.output)


if __name__ == '__main__':
    main()
