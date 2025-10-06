# ============================================================
# omr_service.py - Versión mejorada con mejor detección
# ============================================================

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Configuración por defecto
@dataclass
class OMRConfig:
    target_w: int = 1000
    target_h: int = 1400
    columns: int = 3
    questions_per_column: int = 20
    choices_per_question: int = 4  # A-D
    
    # Calibración de detección
    min_density_threshold: float = 0.30  # Umbral mínimo para considerar marcado
    roi_width_ratio: float = 0.12
    roi_height_ratio: float = 0.6
    
    # Posiciones horizontales de las opciones (ajustables)
    x_ratios: List[float] = None
    
    # Debug
    debug_save: bool = True
    debug_dir: str = "debug_omr"
    
    def __post_init__(self):
        if self.x_ratios is None:
            self.x_ratios = [0.15, 0.37, 0.60, 0.83]

config = OMRConfig()

def _ensure_debug_dir():
    if config.debug_save and not os.path.exists(config.debug_dir):
        os.makedirs(config.debug_dir, exist_ok=True)

def _preprocess(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocesamiento mejorado con múltiples técnicas"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Ecualización de histograma para mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Reducción de ruido
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)
    
    # Blur suave
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Binarización adaptativa
    th_adaptive = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )
    
    # Binarización Otsu como respaldo
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combinar ambos métodos
    th_combined = cv2.bitwise_or(th_adaptive, th_otsu)
    
    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th_clean = cv2.morphologyEx(th_combined, cv2.MORPH_CLOSE, kernel)
    
    return gray, th_clean

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena los 4 puntos en: TL, TR, BR, BL"""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4,2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]       # TL (menor suma)
    rect[2] = pts[np.argmax(s)]       # BR (mayor suma)
    rect[1] = pts[np.argmin(diff)]    # TR (menor diferencia)
    rect[3] = pts[np.argmax(diff)]    # BL (mayor diferencia)
    return rect

def _detect_omr_sheet(th_bin: np.ndarray) -> Optional[np.ndarray]:
    """Detecta el contorno de la hoja OMR"""
    # Dilatación para unir contornos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilated = cv2.dilate(th_bin, kernel, iterations=2)
    
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    
    # Filtrar por área (debe ser el contorno más grande)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for cnt in cnts[:5]:  # Revisar los 5 más grandes
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        
        # Debe ser un área significativa
        if area < 50000:
            continue
            
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            return approx.reshape(4,2).astype(np.float32)
    
    return None

def _warp_to_standard(img_bgr: np.ndarray, th_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Aplica transformación perspectiva a tamaño estándar"""
    corners = _detect_omr_sheet(th_bin)
    
    if corners is not None:
        rect = _order_corners(corners)
        dst = np.array([
            [0, 0],
            [config.target_w-1, 0],
            [config.target_w-1, config.target_h-1],
            [0, config.target_h-1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        img_w = cv2.warpPerspective(img_bgr, M, (config.target_w, config.target_h))
        th_w = cv2.warpPerspective(th_bin, M, (config.target_w, config.target_h))
        return img_w, th_w, True
    else:
        # Fallback: simple resize
        img_w = cv2.resize(img_bgr, (config.target_w, config.target_h))
        th_w = cv2.resize(th_bin, (config.target_w, config.target_h))
        return img_w, th_w, False

def _split_columns(th_bin: np.ndarray) -> List[np.ndarray]:
    """Divide la imagen en columnas"""
    h, w = th_bin.shape
    col_w = w // config.columns
    return [th_bin[:, i*col_w:(i+1)*col_w] for i in range(config.columns)]

def _detect_circle_fill(img_bin: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """Detecta qué tan lleno está un círculo usando contornos"""
    x, y, r = int(cx), int(cy), int(radius)
    
    # Crear máscara circular
    mask = np.zeros_like(img_bin, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    
    # Extraer región
    roi = cv2.bitwise_and(img_bin, mask)
    
    # Calcular densidad
    circle_area = np.count_nonzero(mask)
    if circle_area == 0:
        return 0.0
    
    filled_pixels = np.count_nonzero(roi)
    density = filled_pixels / circle_area
    
    return density

def _density_in_roi(img_bin: np.ndarray, cx: float, cy: float, w: float, h: float) -> float:
    """Calcula densidad en región rectangular"""
    w = int(max(8, w))
    h = int(max(8, h))
    x0, x1 = int(cx - w/2), int(cx + w/2)
    y0, y1 = int(cy - h/2), int(cy + h/2)
    
    H, W = img_bin.shape
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    
    roi = img_bin[y0:y1, x0:x1]
    if roi.size == 0:
        return 0
    
    return np.count_nonzero(roi) / roi.size

def _visualize_detections(img_bgr: np.ndarray, col_bin: np.ndarray, 
                         col_idx: int, detections: List[Dict], save_tag: str):
    """Genera imagen de debug mostrando las detecciones"""
    h, w = col_bin.shape
    vis = cv2.cvtColor(col_bin, cv2.COLOR_GRAY2BGR)
    
    row_height = h / config.questions_per_column
    roi_w = w * config.roi_width_ratio
    roi_h = row_height * config.roi_height_ratio
    
    for det in detections:
        q_idx = det['question_idx']
        cy = (q_idx + 0.5) * row_height
        
        for choice_idx, (xr, density) in enumerate(zip(config.x_ratios, det['densities'])):
            cx = w * xr
            
            # Color según si está marcado
            is_marked = (choice_idx == det['selected'])
            color = (0, 255, 0) if is_marked else (0, 0, 255)
            
            # Dibujar ROI
            x0, x1 = int(cx - roi_w/2), int(cx + roi_w/2)
            y0, y1 = int(cy - roi_h/2), int(cy + roi_h/2)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            
            # Mostrar densidad
            text = f"{density:.2f}"
            cv2.putText(vis, text, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    filename = f"{save_tag}_col{col_idx}_detections.jpg"
    cv2.imwrite(os.path.join(config.debug_dir, filename), vis)

def extract_answers(img_bgr: np.ndarray, save_tag: Optional[str] = None) -> Tuple[List[int], Dict]:
    """
    Extrae respuestas de una hoja OMR
    
    Returns:
        (answers, metadata) donde:
        - answers: lista de índices 0-3 (A-D)
        - metadata: información de debug
    """
    _ensure_debug_dir()
    
    # Preprocesar
    gray, th_w = _preprocess(img_bgr)
    
    # Aplicar transformación perspectiva
    img_w, th_bin, warp_success = _warp_to_standard(img_bgr, th_w)
    
    # Guardar imágenes de debug
    if config.debug_save and save_tag:
        cv2.imwrite(os.path.join(config.debug_dir, f"{save_tag}_0_original.jpg"), img_bgr)
        cv2.imwrite(os.path.join(config.debug_dir, f"{save_tag}_1_gray.jpg"), gray)
        cv2.imwrite(os.path.join(config.debug_dir, f"{save_tag}_2_binary.jpg"), th_bin)
        cv2.imwrite(os.path.join(config.debug_dir, f"{save_tag}_3_warped.jpg"), img_w)
    
    # Dividir en columnas
    cols = _split_columns(th_bin)
    
    answers: List[int] = []
    all_detections = []
    
    for col_idx, col_bin in enumerate(cols):
        h, w = col_bin.shape
        row_height = h / config.questions_per_column
        roi_w = w * config.roi_width_ratio
        roi_h = row_height * config.roi_height_ratio
        
        col_detections = []
        
        for q_idx in range(config.questions_per_column):
            cy = (q_idx + 0.5) * row_height
            densities = []
            
            # Medir densidad en cada opción
            for xr in config.x_ratios:
                cx = w * xr
                density = _density_in_roi(col_bin, cx, cy, roi_w, roi_h)
                densities.append(density)
            
            # Seleccionar la opción con mayor densidad
            max_idx = int(np.argmax(densities))
            max_density = densities[max_idx]
            
            # Validar que supere el umbral
            if max_density < config.min_density_threshold:
                # Si ninguna supera el umbral, considerar sin respuesta (A por defecto)
                selected = 0
            else:
                selected = max_idx
            
            answers.append(selected)
            
            col_detections.append({
                'question_idx': q_idx,
                'densities': densities,
                'selected': selected,
                'confidence': max_density
            })
        
        all_detections.append(col_detections)
        
        # Visualizar detecciones
        if config.debug_save and save_tag:
            _visualize_detections(img_w, col_bin, col_idx, col_detections, save_tag)
    
    # Metadata
    metadata = {
        'warp_success': warp_success,
        'detections': all_detections,
        'avg_confidence': np.mean([d['confidence'] for col in all_detections for d in col])
    }
    
    # Asegurar 60 respuestas
    total_expected = config.columns * config.questions_per_column
    if len(answers) > total_expected:
        answers = answers[:total_expected]
    elif len(answers) < total_expected:
        answers.extend([0] * (total_expected - len(answers)))
    
    return answers, metadata

def compare_answers(teacher: List[int], student: List[int]) -> Dict:
    """Compara respuestas del profesor vs estudiante"""
    n = min(len(teacher), len(student))
    choices_map = "ABCD"
    correct = 0
    detail = []
    
    for i in range(n):
        ok = (teacher[i] == student[i])
        correct += 1 if ok else 0
        detail.append({
            "pregunta": i + 1,
            "correcta": choices_map[teacher[i]] if teacher[i] < len(choices_map) else "?",
            "alumno": choices_map[student[i]] if student[i] < len(choices_map) else "?",
            "acierto": ok
        })
    
    return {
        "total": n,
        "aciertos": correct,
        "porcentaje": round(100.0 * correct / max(n, 1), 2),
        "detalle": detail
    }

def calibrate_from_sample(img_bgr: np.ndarray, expected_answers: List[int]) -> Dict:
    """
    Calibra el sistema usando una imagen de muestra con respuestas conocidas
    
    Args:
        img_bgr: Imagen de la hoja OMR
        expected_answers: Lista de respuestas correctas (0-3 para A-D)
    
    Returns:
        Diccionario con métricas de calibración y sugerencias
    """
    answers, metadata = extract_answers(img_bgr, save_tag="calibration")
    
    # Comparar con respuestas esperadas
    n = min(len(answers), len(expected_answers))
    correct = sum(1 for i in range(n) if answers[i] == expected_answers[i])
    accuracy = correct / n if n > 0 else 0
    
    # Analizar densidades
    all_densities = []
    for col in metadata['detections']:
        for det in col:
            all_densities.extend(det['densities'])
    
    suggestions = {
        'accuracy': accuracy,
        'correct': correct,
        'total': n,
        'avg_density': np.mean(all_densities),
        'std_density': np.std(all_densities),
        'min_density': np.min(all_densities),
        'max_density': np.max(all_densities),
        'warp_success': metadata['warp_success']
    }
    
    # Sugerencias de ajuste
    if accuracy < 0.8:
        suggestions['recommendations'] = []
        if not metadata['warp_success']:
            suggestions['recommendations'].append("La hoja no se detectó correctamente. Asegúrate de que esté completa en la imagen.")
        if suggestions['max_density'] < 0.3:
            suggestions['recommendations'].append("Las marcas son muy tenues. Aumentar contraste o reducir min_density_threshold.")
        if suggestions['std_density'] < 0.1:
            suggestions['recommendations'].append("Poca diferencia entre marcadas y no marcadas. Revisar iluminación.")
    
    return suggestions

def save_config():
    """Guarda la configuración actual"""
    config_dict = {
        'x_ratios': config.x_ratios,
        'min_density_threshold': config.min_density_threshold,
        'roi_width_ratio': config.roi_width_ratio,
        'roi_height_ratio': config.roi_height_ratio
    }
    with open('omr_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config():
    """Carga configuración guardada"""
    try:
        with open('omr_config.json', 'r') as f:
            config_dict = json.load(f)
            config.x_ratios = config_dict['x_ratios']
            config.min_density_threshold = config_dict['min_density_threshold']
            config.roi_width_ratio = config_dict['roi_width_ratio']
            config.roi_height_ratio = config_dict['roi_height_ratio']
    except FileNotFoundError:
        pass  # Usar valores por defecto