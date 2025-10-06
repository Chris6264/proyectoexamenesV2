from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import os
import shutil
from .omr_service import (
    extract_answers, 
    compare_answers, 
    calibrate_from_sample,
    config,
    save_config,
    load_config
)

app = FastAPI(
    title="OMR Grading API",
    description="API para calificar hojas de respuestas OMR",
    version="2.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar configuración al iniciar
load_config()

@app.get("/")
def read_root():
    return {
        "message": "Backend de OMR funcionando correctamente ✅",
        "version": "2.0",
        "endpoints": {
            "grade": "/grade - Calificar examen",
            "calibrate": "/calibrate - Calibrar sistema",
            "config": "/config - Ver/actualizar configuración",
            "debug_images": "/debug/{filename} - Ver imágenes de debug"
        }
    }

@app.post("/grade")
async def grade(
    teacher: UploadFile = File(..., description="Hoja con respuestas correctas"),
    student: UploadFile = File(..., description="Hoja del estudiante")
):
    """
    Califica un examen comparando las respuestas del profesor con las del alumno
    """
    try:
        # Leer imágenes
        teacher_bytes = np.frombuffer(await teacher.read(), np.uint8)
        student_bytes = np.frombuffer(await student.read(), np.uint8)

        teacher_img = cv2.imdecode(teacher_bytes, cv2.IMREAD_COLOR)
        student_img = cv2.imdecode(student_bytes, cv2.IMREAD_COLOR)

        if teacher_img is None or student_img is None:
            raise HTTPException(status_code=400, detail="No se pudieron leer las imágenes")

        # Extraer respuestas
        teacher_ans, teacher_meta = extract_answers(teacher_img, save_tag="profesor")
        student_ans, student_meta = extract_answers(student_img, save_tag="alumno")

        # Comparar
        result = compare_answers(teacher_ans, student_ans)
        
        # Agregar metadata
        result['metadata'] = {
            'teacher': {
                'warp_success': teacher_meta['warp_success'],
                'avg_confidence': round(teacher_meta['avg_confidence'], 3)
            },
            'student': {
                'warp_success': student_meta['warp_success'],
                'avg_confidence': round(student_meta['avg_confidence'], 3)
            }
        }
        
        # Agregar advertencias si hay problemas
        warnings = []
        if not teacher_meta['warp_success']:
            warnings.append("La hoja del profesor no se detectó correctamente")
        if not student_meta['warp_success']:
            warnings.append("La hoja del alumno no se detectó correctamente")
        if teacher_meta['avg_confidence'] < 0.25:
            warnings.append("Confianza baja en detección del profesor")
        if student_meta['avg_confidence'] < 0.25:
            warnings.append("Confianza baja en detección del alumno")
        
        if warnings:
            result['warnings'] = warnings

        return JSONResponse(result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar: {str(e)}")

@app.post("/extract")
async def extract(image: UploadFile = File(...)):
    """
    Extrae las respuestas de una hoja OMR sin comparar
    """
    try:
        img_bytes = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

        answers, metadata = extract_answers(img, save_tag="extract")
        
        choices_map = "ABCD"
        return JSONResponse({
            "answers": [choices_map[ans] for ans in answers],
            "answers_numeric": answers,
            "metadata": {
                'warp_success': metadata['warp_success'],
                'avg_confidence': round(metadata['avg_confidence'], 3)
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al extraer: {str(e)}")

class CalibrationRequest(BaseModel):
    expected_answers: List[int]  # Lista de 60 respuestas esperadas (0-3)

@app.post("/calibrate")
async def calibrate(
    image: UploadFile = File(...),
    expected_answers: str = File(...)  # JSON string con las respuestas
):
    """
    Calibra el sistema usando una hoja de muestra con respuestas conocidas
    Ejemplo de expected_answers: "[0,1,2,3,0,1,2,3,...]" (60 números)
    """
    try:
        import json
        
        # Parsear respuestas esperadas
        expected = json.loads(expected_answers)
        if len(expected) != 60:
            raise HTTPException(
                status_code=400, 
                detail=f"Se esperan 60 respuestas, recibidas {len(expected)}"
            )
        
        # Leer imagen
        img_bytes = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

        # Calibrar
        calibration_result = calibrate_from_sample(img, expected)
        
        return JSONResponse(calibration_result)
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON inválido en expected_answers")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en calibración: {str(e)}")

@app.get("/config")
async def get_config():
    """Obtiene la configuración actual del sistema"""
    return JSONResponse({
        "x_ratios": config.x_ratios,
        "min_density_threshold": config.min_density_threshold,
        "roi_width_ratio": config.roi_width_ratio,
        "roi_height_ratio": config.roi_height_ratio,
        "columns": config.columns,
        "questions_per_column": config.questions_per_column,
        "choices_per_question": config.choices_per_question
    })

class ConfigUpdate(BaseModel):
    x_ratios: Optional[List[float]] = None
    min_density_threshold: Optional[float] = None
    roi_width_ratio: Optional[float] = None
    roi_height_ratio: Optional[float] = None

@app.put("/config")
async def update_config(updates: ConfigUpdate):
    """Actualiza la configuración del sistema"""
    if updates.x_ratios is not None:
        if len(updates.x_ratios) != config.choices_per_question:
            raise HTTPException(
                status_code=400,
                detail=f"x_ratios debe tener {config.choices_per_question} valores"
            )
        config.x_ratios = updates.x_ratios
    
    if updates.min_density_threshold is not None:
        if not 0 <= updates.min_density_threshold <= 1:
            raise HTTPException(status_code=400, detail="min_density_threshold debe estar entre 0 y 1")
        config.min_density_threshold = updates.min_density_threshold
    
    if updates.roi_width_ratio is not None:
        if not 0 < updates.roi_width_ratio < 1:
            raise HTTPException(status_code=400, detail="roi_width_ratio debe estar entre 0 y 1")
        config.roi_width_ratio = updates.roi_width_ratio
    
    if updates.roi_height_ratio is not None:
        if not 0 < updates.roi_height_ratio < 1:
            raise HTTPException(status_code=400, detail="roi_height_ratio debe estar entre 0 y 1")
        config.roi_height_ratio = updates.roi_height_ratio
    
    # Guardar configuración
    save_config()
    
    return JSONResponse({
        "message": "Configuración actualizada exitosamente",
        "config": await get_config()
    })

@app.get("/debug/{filename}")
async def get_debug_image(filename: str):
    """Obtiene una imagen de debug generada"""
    filepath = os.path.join(config.debug_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(filepath)

@app.get("/debug/list")
async def list_debug_images():
    """Lista todas las imágenes de debug disponibles"""
    if not os.path.exists(config.debug_dir):
        return JSONResponse({"images": []})
    
    images = [f for f in os.listdir(config.debug_dir) if f.endswith(('.jpg', '.png'))]
    images.sort(reverse=True)  # Más recientes primero
    
    return JSONResponse({
        "images": images,
        "count": len(images)
    })

@app.delete("/debug")
async def clear_debug_images():
    """Elimina todas las imágenes de debug"""
    if os.path.exists(config.debug_dir):
        shutil.rmtree(config.debug_dir)
        os.makedirs(config.debug_dir)
    return JSONResponse({"message": "Imágenes de debug eliminadas"})

@app.get("/health")
async def health_check():
    """Verifica el estado del servicio"""
    return JSONResponse({
        "status": "healthy",
        "version": "2.0",
        "debug_enabled": config.debug_save
    })