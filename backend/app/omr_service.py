# ============================================================
# omr_service.py - Versión mejorada con detección avanzada
# ============================================================

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración por defecto mejorada
@dataclass
class OMRConfig:
    # Dimensiones objetivo
    target_w: int = 1000
    target_h: int = 1400
    columns: int = 3
    questions_per_column: int = 20
    choices_per_question: int = 4  # A-D
    
    # Calibración de detección mejorada
    min_density_threshold: float = 0.25
    min_pattern_threshold: float = 0.35
    roi_width_ratio: float = 0.12
    roi_height_ratio: float = 0.6
    
    # Posiciones horizontales de las opciones
    x_ratios: List[float] = None
    
    # Parámetros de preprocesamiento
    adaptive_block_size: int = 21
    adaptive_c: int = 10
    morph_kernel_size: int = 3
    denoise_h: float = 10.0
    
    # Debug
    debug_save: bool = True
    debug_dir: str = "debug_omr"
    
    def __post_init__(self):
        if self.x_ratios is None:
            self.x_ratios = [0.15, 0.37, 0.60, 0.83]

config = OMRConfig()

def _ensure_debug_dir():
    """Asegura que el directorio de debug exista"""
    if config.debug_save and not os.path.exists(config.debug_dir):
        os.makedirs(config.debug_dir, exist_ok=True)

def _enhanced_preprocess(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesamiento mejorado con múltiples técnicas y análisis de calidad
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Análisis de calidad de imagen
    brightness = np.mean(gray)
    contrast = np.std(gray)
    logger.info(f"Calidad imagen - Brillo: {brightness:.1f}, Contraste: {contrast:.1f}")
    
    # Ecualización de histograma adaptativa
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    
    # Reducción de ruido adaptativa
    if contrast < 40:  # Imagen de bajo contraste
        denoised = cv2.fastNlMeansDenoising(gray_eq, None, h=config.denoise_h * 1.5)
    else:
        denoised = cv2.fastNlMeansDenoising(gray_eq, None, h=config.denoise_h)
    
    # Blur suave para reducir ruido residual
    blur = cv2.GaussianBlur(denoised, (5, 5), 0)
    
    # Binarización adaptativa múltiple
    th_adaptive = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        config.adaptive_block_size,
        config.adaptive_c
    )
    
    # Binarización Otsu como respaldo
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combinar ambos métodos
    th_combined = cv2.bitwise_or(th_adaptive, th_otsu)
    
    # Limpieza morfológica mejorada
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size))
    th_clean = cv2.morphologyEx(th_combined, cv2.MORPH_CLOSE, kernel)
    th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_OPEN, kernel)
    
    return gray, th_clean, th_combined

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena los 4 puntos en: TL, TR, BR, BL"""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]       # TL (menor suma)
    rect[2] = pts[np.argmax(s)]       # BR (mayor suma)
    rect[1] = pts[np.argmin(diff)]    # TR (menor diferencia)
    rect[3] = pts[np.argmax(diff)]    # BL (mayor diferencia)
    return rect

def _detect_omr_sheet(th_bin: np.ndarray) -> Optional[np.ndarray]:
    """Detecta el contorno de la hoja OMR con mejor precisión"""
    # Dilatación para unir contornos cercanos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(th_bin, kernel, iterations=3)
    
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        logger.warning("No se encontraron contornos")
        return None
    
    # Filtrar por área y relación de aspecto
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for cnt in cnts[:5]:  # Revisar los 5 más grandes
        area = cv2.contourArea(cnt)
        
        # Filtrar por área mínima
        if area < 50000:
            continue
            
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            # Verificar relación de aspecto aproximada de hoja A4
            corners = approx.reshape(4, 2).astype(np.float32)
            rect = _order_corners(corners)
            
            width_a = np.linalg.norm(rect[0] - rect[1])
            width_b = np.linalg.norm(rect[2] - rect[3])
            height_a = np.linalg.norm(rect[0] - rect[3])
            height_b = np.linalg.norm(rect[1] - rect[2])
            
            avg_width = (width_a + width_b) / 2
            avg_height = (height_a + height_b) / 2
            
            aspect_ratio = avg_width / avg_height
            expected_ratio = config.target_w / config.target_h
            
            if 0.5 <= aspect_ratio / expected_ratio <= 2.0:
                logger.info(f"Hoja detectada - Área: {area:.0f}, Relación aspecto: {aspect_ratio:.2f}")
                return corners
    
    logger.warning("No se encontró contorno válido de hoja OMR")
    return None

def _warp_to_standard(img_bgr: np.ndarray, th_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Aplica transformación perspectiva a tamaño estándar con validación"""
    corners = _detect_omr_sheet(th_bin)
    
    if corners is not None:
        rect = _order_corners(corners)
        dst = np.array([
            [0, 0],
            [config.target_w - 1, 0],
            [config.target_w - 1, config.target_h - 1],
            [0, config.target_h - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(rect, dst)
        img_warped = cv2.warpPerspective(img_bgr, M, (config.target_w, config.target_h))
        th_warped = cv2.warpPerspective(th_bin, M, (config.target_w, config.target_h))
        
        # Validar resultado del warp
        if np.mean(th_warped) > 240:  # Imagen mayormente blanca (posible error)
            logger.warning("Warp posiblemente fallido - imagen muy blanca")
            return img_warped, th_warped, False
            
        logger.info("Transformación perspectiva aplicada exitosamente")
        return img_warped, th_warped, True
    else:
        # Fallback: redimensionado simple
        logger.warning("Usando resize como fallback")
        img_resized = cv2.resize(img_bgr, (config.target_w, config.target_h))
        th_resized = cv2.resize(th_bin, (config.target_w, config.target_h))
        return img_resized, th_resized, False

def _split_columns(th_bin: np.ndarray) -> List[np.ndarray]:
    """Divide la imagen en columnas con superposición mínima"""
    h, w = th_bin.shape
    col_w = w // config.columns
    
    columns = []
    for i in range(config.columns):
        start_x = i * col_w
        end_x = (i + 1) * col_w
        
        # Pequeña superposición para evitar perder bordes
        if i > 0:
            start_x -= 2
        if i < config.columns - 1:
            end_x += 2
            
        start_x = max(0, start_x)
        end_x = min(w, end_x)
        
        col = th_bin[:, start_x:end_x]
        columns.append(col)
    
    return columns

def _enhanced_mark_detection(img_bin: np.ndarray, cx: float, cy: float, w: float, h: float) -> Dict[str, float]:
    """
    Detección mejorada de marcas usando múltiples técnicas
    """
    w = int(max(10, w))
    h = int(max(10, h))
    x0, x1 = int(cx - w/2), int(cx + w/2)
    y0, y1 = int(cy - h/2), int(cy + h/2)
    
    H, W = img_bin.shape
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    
    roi = img_bin[y0:y1, x0:x1]
    if roi.size == 0:
        return {"density": 0.0, "confidence": 0.0, "pattern_score": 0.0, "circle_score": 0.0}
    
    # 1. Densidad básica
    density = np.count_nonzero(roi) / roi.size
    
    # 2. Detección de patrones circulares
    circle_score = 0.0
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            circle_score = min(circularity, 1.0)
    
    # 3. Análisis de distribución espacial
    center_x, center_y = roi.shape[1] // 2, roi.shape[0] // 2
    center_mask = np.zeros_like(roi)
    cv2.circle(center_mask, (center_x, center_y), min(center_x, center_y) // 2, 255, -1)
    center_density = np.count_nonzero(cv2.bitwise_and(roi, center_mask)) / np.count_nonzero(center_mask) if np.count_nonzero(center_mask) > 0 else 0
    
    # 4. Puntuación combinada con pesos optimizados
    pattern_score = (0.5 * density + 
                    0.3 * circle_score + 
                    0.2 * center_density)
    
    # 5. Cálculo de confianza
    confidence = min(pattern_score * 1.2, 1.0)  # Escalar ligeramente
    
    return {
        "density": float(density),
        "circle_score": float(circle_score),
        "center_density": float(center_density),
        "pattern_score": float(pattern_score),
        "confidence": float(confidence)
    }

def _visualize_detections(img_bgr: np.ndarray, col_bin: np.ndarray, 
                         col_idx: int, detections: List[Dict], save_tag: str):
    """Genera imagen de debug mostrando las detecciones con información mejorada"""
    h, w = col_bin.shape
    vis = cv2.cvtColor(col_bin, cv2.COLOR_GRAY2BGR)
    
    row_height = h / config.questions_per_column
    roi_w = w * config.roi_width_ratio
    roi_h = row_height * config.roi_height_ratio
    
    for det in detections:
        q_idx = det['question_idx']
        cy = (q_idx + 0.5) * row_height
        
        for choice_idx, detection_data in enumerate(det['detections']):
            xr = config.x_ratios[choice_idx]
            cx = w * xr
            
            # Color según si está marcado
            is_marked = (choice_idx == det['selected'])
            color = (0, 255, 0) if is_marked else (0, 0, 255)
            thickness = 3 if is_marked else 1
            
            # Dibujar ROI
            x0, x1 = int(cx - roi_w/2), int(cx + roi_w/2)
            y0, y1 = int(cy - roi_h/2), int(cy + roi_h/2)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness)
            
            # Mostrar información de detección
            text = f"P:{detection_data['pattern_score']:.2f}"
            cv2.putText(vis, text, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Dibujar punto central
            cv2.circle(vis, (int(cx), int(cy)), 3, color, -1)
    
    # Guardar imagen
    filename = f"{save_tag}_col{col_idx}_detections.jpg"
    cv2.imwrite(os.path.join(config.debug_dir, filename), vis)
    logger.info(f"Imagen de debug guardada: {filename}")

def extract_answers(img_bgr: np.ndarray, save_tag: Optional[str] = None) -> Tuple[List[int], Dict]:
    """
    Extrae respuestas de una hoja OMR con detección mejorada
    
    Returns:
        (answers, metadata) donde:
        - answers: lista de índices 0-3 (A-D)
        - metadata: información de debug y calidad
    """
    _ensure_debug_dir()
    
    try:
        # Preprocesamiento mejorado
        gray, th_bin, th_combined = _enhanced_preprocess(img_bgr)
        
        # Aplicar transformación perspectiva
        img_warped, th_warped, warp_success = _warp_to_standard(img_bgr, th_bin)
        
        # Guardar imágenes de debug
        if config.debug_save and save_tag:
            cv2.imwrite(os.path.join(config.debug_dir, f"{save_tag}_0_original.jpg"), img_bgr)
            cv2.imwrite(os.path.join(config.debug_dir, f"{save_tag}_1_gray.jpg"), gray)
            cv2.imwrite(os.path.join(config.debug_dir, f"{save_tag}_2_binary.jpg"), th_bin)
            cv2.imwrite(os.path.join(config.debug_dir, f"{save_tag}_3_warped.jpg"), img_warped)
            cv2.imwrite(os.path.join(config.debug_dir, f"{save_tag}_4_th_clean.jpg"), th_warped)
        
        # Dividir en columnas
        cols = _split_columns(th_warped)
        
        answers: List[int] = []
        all_detections = []
        confidence_scores = []
        
        for col_idx, col_bin in enumerate(cols):
            h, w = col_bin.shape
            row_height = h / config.questions_per_column
            roi_w = w * config.roi_width_ratio
            roi_h = row_height * config.roi_height_ratio
            
            col_detections = []
            
            for q_idx in range(config.questions_per_column):
                cy = (q_idx + 0.5) * row_height
                detections_data = []
                pattern_scores = []
                
                # Medir en cada opción usando detección mejorada
                for xr in config.x_ratios:
                    cx = w * xr
                    detection_result = _enhanced_mark_detection(col_bin, cx, cy, roi_w, roi_h)
                    detections_data.append(detection_result)
                    pattern_scores.append(detection_result['pattern_score'])
                
                # Seleccionar la opción con mayor pattern_score
                max_idx = int(np.argmax(pattern_scores))
                max_pattern_score = pattern_scores[max_idx]
                max_confidence = detections_data[max_idx]['confidence']
                
                # Validación mejorada con múltiples umbrales
                if (max_pattern_score < config.min_pattern_threshold or 
                    detections_data[max_idx]['density'] < config.min_density_threshold):
                    selected = -1  # Sin respuesta
                    confidence = 0.0
                else:
                    selected = max_idx
                    confidence = max_confidence
                
                answers.append(selected)
                confidence_scores.append(confidence)
                
                col_detections.append({
                    'question_idx': q_idx,
                    'detections': detections_data,
                    'pattern_scores': pattern_scores,
                    'selected': selected,
                    'confidence': confidence,
                    'max_pattern_score': max_pattern_score
                })
            
            all_detections.append(col_detections)
            
            # Visualizar detecciones
            if config.debug_save and save_tag:
                _visualize_detections(img_warped, col_bin, col_idx, col_detections, save_tag)
        
        # Calcular métricas de calidad
        valid_answers = [ans for ans in answers if ans != -1]
        answer_quality = len(valid_answers) / len(answers) if answers else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        logger.info(f"Extracción completada - Respuestas válidas: {len(valid_answers)}/{len(answers)}, "
                   f"Calidad: {answer_quality:.2f}, Confianza promedio: {avg_confidence:.2f}")
        
        # Metadata mejorada
        metadata = {
            'warp_success': warp_success,
            'detections': all_detections,
            'answer_quality': answer_quality,
            'avg_confidence': avg_confidence,
            'confidence_scores': confidence_scores,
            'total_questions': len(answers),
            'valid_answers': len(valid_answers)
        }
        
        return answers, metadata
        
    except Exception as e:
        logger.error(f"Error en extract_answers: {str(e)}")
        # Retornar valores por defecto en caso de error
        default_answers = [0] * (config.columns * config.questions_per_column)
        return default_answers, {'error': str(e), 'warp_success': False, 'avg_confidence': 0}

def compare_answers(teacher: List[int], student: List[int]) -> Dict:
    """Compara respuestas del profesor vs estudiante con análisis detallado"""
    n = min(len(teacher), len(student))
    choices_map = "ABCD"
    
    correct = 0
    detail = []
    analysis = {
        "A": 0, "B": 0, "C": 0, "D": 0,
        "empty_teacher": 0,
        "empty_student": 0
    }
    
    for i in range(n):
        teacher_ans = teacher[i]
        student_ans = student[i]
        
        # Contar distribuciones
        if 0 <= student_ans < len(choices_map):
            analysis[choices_map[student_ans]] += 1
        
        if teacher_ans == -1:
            analysis["empty_teacher"] += 1
        if student_ans == -1:
            analysis["empty_student"] += 1
        
        # Comparar respuestas (ignorar si teacher no tiene respuesta)
        if teacher_ans != -1:
            ok = (teacher_ans == student_ans)
            correct += 1 if ok else 0
        else:
            ok = None  # No calificable
        
        detail.append({
            "pregunta": i + 1,
            "correcta": choices_map[teacher_ans] if 0 <= teacher_ans < len(choices_map) else "X",
            "alumno": choices_map[student_ans] if 0 <= student_ans < len(choices_map) else "X",
            "acierto": ok
        })
    
    total_comparable = n - analysis["empty_teacher"]
    percentage = round(100.0 * correct / max(total_comparable, 1), 2) if total_comparable > 0 else 0
    
    return {
        "total_preguntas": n,
        "total_comparables": total_comparable,
        "aciertos": correct,
        "porcentaje": percentage,
        "detalle": detail,
        "analisis_respuestas": analysis
    }

def calibrate_from_sample(img_bgr: np.ndarray, expected_answers: List[int]) -> Dict:
    """
    Calibra el sistema usando una imagen de muestra con respuestas conocidas
    """
    answers, metadata = extract_answers(img_bgr, save_tag="calibration")
    
    # Comparar con respuestas esperadas
    n = min(len(answers), len(expected_answers))
    correct = 0
    detailed_comparison = []
    
    for i in range(n):
        is_correct = (answers[i] == expected_answers[i])
        correct += 1 if is_correct else 0
        detailed_comparison.append({
            "question": i + 1,
            "expected": expected_answers[i],
            "detected": answers[i],
            "correct": is_correct,
            "confidence": metadata['confidence_scores'][i] if i < len(metadata['confidence_scores']) else 0
        })
    
    accuracy = correct / n if n > 0 else 0
    
    # Analizar métricas de detección
    all_pattern_scores = []
    all_densities = []
    
    for col in metadata['detections']:
        for det in col:
            for detection in det['detections']:
                all_pattern_scores.append(detection['pattern_score'])
                all_densities.append(detection['density'])
    
    calibration_result = {
        'accuracy': accuracy,
        'correct': correct,
        'total': n,
        'warp_success': metadata['warp_success'],
        'answer_quality': metadata['answer_quality'],
        'avg_confidence': metadata['avg_confidence'],
        'pattern_score_stats': {
            'mean': float(np.mean(all_pattern_scores)),
            'std': float(np.std(all_pattern_scores)),
            'min': float(np.min(all_pattern_scores)),
            'max': float(np.max(all_pattern_scores))
        },
        'density_stats': {
            'mean': float(np.mean(all_densities)),
            'std': float(np.std(all_densities)),
            'min': float(np.min(all_densities)),
            'max': float(np.max(all_densities))
        },
        'detailed_comparison': detailed_comparison
    }
    
    # Generar recomendaciones
    recommendations = []
    
    if accuracy < 0.9:
        if not metadata['warp_success']:
            recommendations.append("❌ La hoja no se detectó correctamente. Mejora la iluminación y el encuadre.")
        
        if calibration_result['pattern_score_stats']['max'] < 0.4:
            recommendations.append("🔍 Las marcas son muy tenues. Usa lápiz más oscuro o ajusta min_density_threshold.")
        
        if calibration_result['pattern_score_stats']['std'] < 0.1:
            recommendations.append("⚡ Poca diferencia entre marcadas/no marcadas. Revisa el contraste.")
    
    if recommendations:
        calibration_result['recommendations'] = recommendations
    
    logger.info(f"Calibración completada - Precisión: {accuracy:.2f}, Recomendaciones: {len(recommendations)}")
    return calibration_result

def save_config():
    """Guarda la configuración actual en archivo JSON"""
    config_dict = {
        'x_ratios': config.x_ratios,
        'min_density_threshold': config.min_density_threshold,
        'min_pattern_threshold': config.min_pattern_threshold,
        'roi_width_ratio': config.roi_width_ratio,
        'roi_height_ratio': config.roi_height_ratio,
        'adaptive_block_size': config.adaptive_block_size,
        'adaptive_c': config.adaptive_c
    }
    with open('omr_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Configuración guardada en omr_config.json")

def load_config():
    """Carga configuración desde archivo JSON"""
    try:
        with open('omr_config.json', 'r') as f:
            config_dict = json.load(f)
            config.x_ratios = config_dict.get('x_ratios', config.x_ratios)
            config.min_density_threshold = config_dict.get('min_density_threshold', config.min_density_threshold)
            config.min_pattern_threshold = config_dict.get('min_pattern_threshold', config.min_pattern_threshold)
            config.roi_width_ratio = config_dict.get('roi_width_ratio', config.roi_width_ratio)
            config.roi_height_ratio = config_dict.get('roi_height_ratio', config.roi_height_ratio)
            config.adaptive_block_size = config_dict.get('adaptive_block_size', config.adaptive_block_size)
            config.adaptive_c = config_dict.get('adaptive_c', config.adaptive_c)
        logger.info("Configuración cargada desde omr_config.json")
    except FileNotFoundError:
        logger.info("No se encontró archivo de configuración, usando valores por defecto")
    except Exception as e:
        logger.warning(f"Error cargando configuración: {e}, usando valores por defecto")