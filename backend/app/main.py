# ============================================================
# main.py - API FastAPI para sistema OMR
# ============================================================

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import cv2
import os
import shutil
import json
import uuid
from datetime import datetime

# Importar el servicio OMR
from omr_service import (
    extract_answers, 
    compare_answers, 
    calibrate_from_sample,
    config,
    save_config,
    load_config
)

# Inicializar FastAPI
app = FastAPI(
    title="OMR Grading API Pro",
    description="API profesional para corrección automática de exámenes OMR",
    version="2.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar directorio de debug para acceso web
app.mount("/debug", StaticFiles(directory=config.debug_dir), name="debug")

# Cargar configuración al iniciar
load_config()

# Modelos Pydantic
class ConfigUpdate(BaseModel):
    x_ratios: Optional[List[float]] = None
    min_density_threshold: Optional[float] = None
    min_pattern_threshold: Optional[float] = None
    roi_width_ratio: Optional[float] = None
    roi_height_ratio: Optional[float] = None
    adaptive_block_size: Optional[int] = None
    adaptive_c: Optional[int] = None

class BatchGradeRequest(BaseModel):
    teacher_id: str
    student_files: List[str]  # Lista de nombres de archivo

@app.get("/")
async def read_root():
    """Endpoint raíz con información del API"""
    return {
        "message": "🚀 Backend de OMR Pro funcionando correctamente",
        "version": "2.1",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "grade": "POST /grade - Calificar examen individual",
            "batch_grade": "POST /batch-grade - Calificación por lote",
            "extract": "POST /extract - Extraer respuestas de hoja",
            "calibrate": "POST /calibrate - Calibrar sistema",
            "config": "GET/PUT /config - Configuración del sistema",
            "debug": "GET /debug/list - Ver imágenes de diagnóstico",
            "health": "GET /health - Estado del servicio"
        }
    }

@app.post("/grade")
async def grade_exam(
    teacher: UploadFile = File(..., description="Hoja del profesor con respuestas correctas"),
    student: UploadFile = File(..., description="Hoja del estudiante a calificar")
):
    """
    Califica un examen comparando las respuestas del profesor con las del alumno
    """
    try:
        # Generar ID único para esta sesión
        session_id = str(uuid.uuid4())[:8]
        
        # Validar tipos de archivo
        if not teacher.content_type.startswith('image/') or not student.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos de imagen")
        
        # Leer y decodificar imágenes
        teacher_bytes = np.frombuffer(await teacher.read(), np.uint8)
        student_bytes = np.frombuffer(await student.read(), np.uint8)

        teacher_img = cv2.imdecode(teacher_bytes, cv2.IMREAD_COLOR)
        student_img = cv2.imdecode(student_bytes, cv2.IMREAD_COLOR)

        if teacher_img is None or student_img is None:
            raise HTTPException(status_code=400, detail="No se pudieron decodificar las imágenes")

        # Extraer respuestas
        teacher_ans, teacher_meta = extract_answers(teacher_img, save_tag=f"t_{session_id}")
        student_ans, student_meta = extract_answers(student_img, save_tag=f"s_{session_id}")

        # Comparar respuestas
        result = compare_answers(teacher_ans, student_ans)
        
        # Enriquecer respuesta con metadata
        result['session_id'] = session_id
        result['metadata'] = {
            'teacher': {
                'warp_success': teacher_meta['warp_success'],
                'answer_quality': round(teacher_meta.get('answer_quality', 0), 3),
                'avg_confidence': round(teacher_meta.get('avg_confidence', 0), 3),
                'valid_answers': teacher_meta.get('valid_answers', 0)
            },
            'student': {
                'warp_success': student_meta['warp_success'],
                'answer_quality': round(student_meta.get('answer_quality', 0), 3),
                'avg_confidence': round(student_meta.get('avg_confidence', 0), 3),
                'valid_answers': student_meta.get('valid_answers', 0)
            }
        }
        
        # Agregar advertencias si es necesario
        warnings = []
        if not teacher_meta['warp_success']:
            warnings.append("⚠️ La hoja del profesor no se alineó correctamente")
        if not student_meta['warp_success']:
            warnings.append("⚠️ La hoja del alumno no se alineó correctamente")
        if teacher_meta.get('answer_quality', 1) < 0.8:
            warnings.append("🔍 Baja calidad en respuestas detectadas del profesor")
        if student_meta.get('answer_quality', 1) < 0.8:
            warnings.append("🔍 Baja calidad en respuestas detectadas del alumno")
        
        if warnings:
            result['warnings'] = warnings

        return JSONResponse(result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al procesar: {str(e)}")

@app.post("/extract")
async def extract_answers_endpoint(
    image: UploadFile = File(..., description="Hoja OMR para extraer respuestas")
):
    """
    Extrae las respuestas de una hoja OMR sin comparar
    """
    try:
        # Validar tipo de archivo
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos de imagen")
        
        # Leer y decodificar imagen
        img_bytes = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

        # Extraer respuestas
        session_id = str(uuid.uuid4())[:8]
        answers, metadata = extract_answers(img, save_tag=f"extract_{session_id}")
        
        # Mapear a letras
        choices_map = "ABCD"
        answers_letters = [choices_map[ans] if 0 <= ans < len(choices_map) else "X" for ans in answers]
        
        return JSONResponse({
            "session_id": session_id,
            "answers": answers_letters,
            "answers_numeric": answers,
            "metadata": {
                'warp_success': metadata['warp_success'],
                'answer_quality': round(metadata.get('answer_quality', 0), 3),
                'avg_confidence': round(metadata.get('avg_confidence', 0), 3),
                'valid_answers': metadata.get('valid_answers', 0),
                'total_questions': metadata.get('total_questions', 0)
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al extraer respuestas: {str(e)}")

@app.post("/calibrate")
async def calibrate_system(
    image: UploadFile = File(..., description="Hoja de calibración con respuestas conocidas"),
    expected_answers: str = Form(..., description='JSON array con 60 respuestas esperadas ej: [0,1,2,3,...]')
):
    """
    Calibra el sistema usando una hoja de muestra con respuestas conocidas
    """
    try:
        # Parsear respuestas esperadas
        expected = json.loads(expected_answers)
        if len(expected) != 60:
            raise HTTPException(
                status_code=400, 
                detail=f"Se esperaban 60 respuestas, se recibieron {len(expected)}"
            )
        
        # Validar que las respuestas estén en rango
        if any(ans not in [0, 1, 2, 3] for ans in expected):
            raise HTTPException(
                status_code=400,
                detail="Las respuestas deben ser valores entre 0-3 (A-D)"
            )
        
        # Leer y decodificar imagen
        img_bytes = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

        # Ejecutar calibración
        session_id = str(uuid.uuid4())[:8]
        calibration_result = calibrate_from_sample(img, expected)
        calibration_result['session_id'] = session_id
        
        return JSONResponse(calibration_result)
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Formato JSON inválido en expected_answers")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en calibración: {str(e)}")

@app.get("/config")
async def get_configuration():
    """Obtiene la configuración actual del sistema OMR"""
    return JSONResponse({
        "geometry": {
            "target_width": config.target_w,
            "target_height": config.target_h,
            "columns": config.columns,
            "questions_per_column": config.questions_per_column,
            "choices_per_question": config.choices_per_question
        },
        "detection": {
            "x_ratios": config.x_ratios,
            "min_density_threshold": config.min_density_threshold,
            "min_pattern_threshold": config.min_pattern_threshold,
            "roi_width_ratio": config.roi_width_ratio,
            "roi_height_ratio": config.roi_height_ratio
        },
        "processing": {
            "adaptive_block_size": config.adaptive_block_size,
            "adaptive_c": config.adaptive_c,
            "morph_kernel_size": config.morph_kernel_size,
            "denoise_h": config.denoise_h
        },
        "debug": {
            "debug_save": config.debug_save,
            "debug_dir": config.debug_dir
        }
    })

@app.put("/config")
async def update_configuration(updates: ConfigUpdate):
    """Actualiza la configuración del sistema OMR"""
    try:
        # Validar y actualizar x_ratios
        if updates.x_ratios is not None:
            if len(updates.x_ratios) != config.choices_per_question:
                raise HTTPException(
                    status_code=400,
                    detail=f"x_ratios debe tener exactamente {config.choices_per_question} valores"
                )
            if any(not 0 <= x <= 1 for x in updates.x_ratios):
                raise HTTPException(
                    status_code=400,
                    detail="Todos los valores en x_ratios deben estar entre 0 y 1"
                )
            config.x_ratios = updates.x_ratios
        
        # Actualizar otros parámetros con validación
        if updates.min_density_threshold is not None:
            if not 0 <= updates.min_density_threshold <= 1:
                raise HTTPException(status_code=400, detail="min_density_threshold debe estar entre 0 y 1")
            config.min_density_threshold = updates.min_density_threshold
        
        if updates.min_pattern_threshold is not None:
            if not 0 <= updates.min_pattern_threshold <= 1:
                raise HTTPException(status_code=400, detail="min_pattern_threshold debe estar entre 0 y 1")
            config.min_pattern_threshold = updates.min_pattern_threshold
        
        if updates.roi_width_ratio is not None:
            if not 0.01 <= updates.roi_width_ratio <= 0.5:
                raise HTTPException(status_code=400, detail="roi_width_ratio debe estar entre 0.01 y 0.5")
            config.roi_width_ratio = updates.roi_width_ratio
        
        if updates.roi_height_ratio is not None:
            if not 0.01 <= updates.roi_height_ratio <= 0.8:
                raise HTTPException(status_code=400, detail="roi_height_ratio debe estar entre 0.01 y 0.8")
            config.roi_height_ratio = updates.roi_height_ratio
        
        if updates.adaptive_block_size is not None:
            if updates.adaptive_block_size % 2 == 0 or updates.adaptive_block_size < 3:
                raise HTTPException(status_code=400, detail="adaptive_block_size debe ser impar y ≥ 3")
            config.adaptive_block_size = updates.adaptive_block_size
        
        if updates.adaptive_c is not None:
            config.adaptive_c = updates.adaptive_c
    
        # Guardar configuración
        save_config()
        
        return JSONResponse({
            "message": "✅ Configuración actualizada exitosamente",
            "config": await get_configuration()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando configuración: {str(e)}")

@app.get("/debug/list")
async def list_debug_images():
    """Lista todas las imágenes de diagnóstico disponibles"""
    try:
        if not os.path.exists(config.debug_dir):
            return JSONResponse({"images": [], "count": 0})
        
        images = []
        for f in os.listdir(config.debug_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(config.debug_dir, f)
                stat = os.stat(file_path)
                images.append({
                    "filename": f,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "url": f"/debug/{f}"
                })
        
        # Ordenar por fecha de modificación (más recientes primero)
        images.sort(key=lambda x: x["modified"], reverse=True)
        
        return JSONResponse({
            "images": images,
            "count": len(images),
            "total_size": sum(img["size"] for img in images)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando imágenes: {str(e)}")

@app.delete("/debug")
async def clear_debug_images():
    """Elimina todas las imágenes de diagnóstico"""
    try:
        if os.path.exists(config.debug_dir):
            shutil.rmtree(config.debug_dir)
            os.makedirs(config.debug_dir)
        return JSONResponse({
            "message": "✅ Todas las imágenes de diagnóstico han sido eliminadas",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando imágenes: {str(e)}")

@app.get("/health")
async def health_check():
    """Verifica el estado del servicio y componentes"""
    try:
        # Verificar directorio de debug
        debug_status = os.path.exists(config.debug_dir)
        
        # Verificar que OpenCV funcione
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(config.debug_dir, "health_check.jpg"), test_img)
        opencv_ok = os.path.exists(os.path.join(config.debug_dir, "health_check.jpg"))
        
        # Limpiar archivo de prueba
        if opencv_ok:
            os.remove(os.path.join(config.debug_dir, "health_check.jpg"))
        
        return JSONResponse({
            "status": "healthy",
            "version": "2.1",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "opencv": opencv_ok,
                "debug_directory": debug_status,
                "debug_enabled": config.debug_save
            },
            "system": {
                "python_version": "3.8+",
                "framework": "FastAPI",
                "module": "OMR Pro"
            }
        })
    
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status_code=500)

@app.post("/batch-grade")
async def batch_grade_exams(
    teacher: UploadFile = File(..., description="Hoja del profesor"),
    students: List[UploadFile] = File(..., description="Hojas de múltiples estudiantes")
):
    """
    Califica exámenes por lote para múltiples estudiantes
    """
    try:
        # Procesar hoja del profesor primero
        teacher_bytes = np.frombuffer(await teacher.read(), np.uint8)
        teacher_img = cv2.imdecode(teacher_bytes, cv2.IMREAD_COLOR)
        
        if teacher_img is None:
            raise HTTPException(status_code=400, detail="No se pudo procesar la hoja del profesor")
        
        teacher_ans, teacher_meta = extract_answers(teacher_img, save_tag="batch_teacher")
        
        results = []
        session_id = str(uuid.uuid4())[:8]
        
        # Procesar cada estudiante
        for i, student_file in enumerate(students):
            try:
                student_bytes = np.frombuffer(await student_file.read(), np.uint8)
                student_img = cv2.imdecode(student_bytes, cv2.IMREAD_COLOR)
                
                if student_img is None:
                    results.append({
                        "student": student_file.filename,
                        "error": "No se pudo decodificar la imagen",
                        "success": False
                    })
                    continue
                
                student_ans, student_meta = extract_answers(student_img, save_tag=f"batch_s_{session_id}_{i}")
                comparison = compare_answers(teacher_ans, student_ans)
                
                results.append({
                    "student": student_file.filename,
                    "success": True,
                    "result": comparison,
                    "metadata": {
                        "warp_success": student_meta['warp_success'],
                        "answer_quality": student_meta.get('answer_quality', 0),
                        "avg_confidence": student_meta.get('avg_confidence', 0)
                    }
                })
                
            except Exception as e:
                results.append({
                    "student": student_file.filename,
                    "error": str(e),
                    "success": False
                })
        
        return JSONResponse({
            "session_id": session_id,
            "teacher_processed": True,
            "teacher_metadata": {
                "warp_success": teacher_meta['warp_success'],
                "answer_quality": teacher_meta.get('answer_quality', 0)
            },
            "students_processed": len([r for r in results if r['success']]),
            "students_failed": len([r for r in results if not r['success']]),
            "results": results
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en procesamiento por lote: {str(e)}")

# Ejecutar la aplicación
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)