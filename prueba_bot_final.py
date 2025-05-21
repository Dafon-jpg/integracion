"""
Script unificado para renombrar PDFs de facturas y generar archivos TXT para obras sociales.
Con integración de IA (LayoutLMv3) para mejorar el procesamiento.
"""

import os
import csv
import re
import io
import logging
from datetime import datetime

import pandas as pd
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# Importaciones para IA
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification, LayoutLMv3ForTokenClassification
# from transformers import AdamW
import numpy as np
import cv2
from tqdm import tqdm

# ========== CONFIGURACIÓN ==========
# Ajusta estas rutas según tu configuración
FOLDER_ID = '1-pMVR5Nh4k_Jlenygaju0R1nH0YcGTFP'  # Carpeta con PDFs
SCOPES = ['https://www.googleapis.com/auth/drive']

# Configurar OCR
TESSERACT_CMD = 'tesseract'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("facturas_bot.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configuraciones específicas para generación de TXT
PATTERN_RENAMED = r'^[0-9]{11}_[1-9]\d*_[1-9]\d*_[1-9]\d*\.pdf$'
CODIGO_CBTE = {
    ("FACTURA", "B"): "03",
    ("RECIBO", "B"): "04",
    ("FACTURA", "C"): "05",
    ("RECIBO", "C"): "06",
}
ACTIVIDADES_DEP = {
    "001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
    "011", "012", "037", "038", "039", "040", "041", "042", "043", "044",
    "045", "046", "047", "048", "058", "059", "060", "061", "062", "063",
    "064", "065", "066", "067", "068", "069", "070", "071", "072", "076",
    "077", "078", "096",
}

# ========== CONFIGURACIÓN PARA IA ==========
# Configurar el dispositivo para usar GPU si está disponible (cambia a "cpu" para forzar CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Ruta al modelo fine-tuneado - ¡MODIFICADO!
MODEL_PATH = "./layoutlmv3-finetuned-facturas_final"  # Ruta al modelo fine-tuneado
MAX_LENGTH = 512  # Máxima longitud de tokens para procesar
BATCH_SIZE = 1  # Ajustar según la memoria de GPU

# Verificar GPU
def check_gpu():
    if torch.cuda.is_available():
        logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Memoria GPU disponible: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        logger.info("GPU no disponible, usando CPU")

# ========== CLASES Y FUNCIONES PARA IA ==========
class FacturaProcessor:
    """Clase para procesar facturas con LayoutLMv3"""
    
    def __init__(self, model_path=MODEL_PATH):
        """Inicializa el procesador y los modelos"""
        logger.info(f"Inicializando modelo LayoutLMv3 desde {model_path}...")
        
        try:
            # Intentar cargar el procesador entrenado
            self.processor = LayoutLMv3Processor.from_pretrained(model_path)
            logger.info("Procesador entrenado cargado correctamente")
        except Exception as e:
            logger.warning(f"No se pudo cargar el procesador entrenado: {e}")
            # Cargar procesador base como respaldo
            self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
            logger.info("Procesador base cargado como respaldo")
        
        # Cargar modelo para clasificación de tipo de documento
        self.doc_classifier = None  # Se cargará bajo demanda para ahorrar memoria
        
        # Cargar modelo para extraer entidades
        self.entity_extractor = None  # Se cargará bajo demanda para ahorrar memoria
        
        # Lista de entidades que queremos extraer
        self.entity_labels = [
            "CUIT_PRESTADOR", "CUIT_AFILIADO", "NOMBRE_AFILIADO", "NOMBRE_PRESTADOR",
            "TIPO_FACTURA", "LETRA_FACTURA", "PUNTO_VENTA", "NUMERO_FACTURA",
            "FECHA_EMISION", "CAE", "IMPORTE", "PERIODO", "ACTIVIDAD", "DNI_AFILIADO"
        ]
        
        logger.info("Modelo LayoutLMv3 inicializado correctamente")
    
    def load_doc_classifier(self):
        """Carga el modelo de clasificación de documentos bajo demanda"""
        if self.doc_classifier is None:
            logger.info("Cargando modelo de clasificación de documentos...")
            self.doc_classifier = LayoutLMv3ForSequenceClassification.from_pretrained(
                MODEL_PATH if os.path.exists(MODEL_PATH) else "microsoft/layoutlmv3-base", 
                num_labels=2  # Factura válida o no válida
            )
            self.doc_classifier.to(DEVICE)
            logger.info("Modelo de clasificación cargado")
    
    def load_entity_extractor(self):
        """Carga el modelo de extracción de entidades bajo demanda"""
        if self.entity_extractor is None:
            logger.info("Cargando modelo de extracción de entidades fine-tuneado...")
            
            try:
                # Intenta cargar el modelo fine-tuneado
                self.entity_extractor = LayoutLMv3ForTokenClassification.from_pretrained(
                    MODEL_PATH
                )
                self.entity_extractor.to(DEVICE)
                logger.info("Modelo de extracción de entidades fine-tuneado cargado correctamente")
            except Exception as e:
                logger.error(f"Error al cargar el modelo fine-tuneado: {e}")
                logger.info("Intentando cargar el modelo base como respaldo...")
                # Carga modelo base como respaldo
                self.entity_extractor = LayoutLMv3ForTokenClassification.from_pretrained(
                    "microsoft/layoutlmv3-base",
                    num_labels=len(self.entity_labels) * 2 + 1  # B-tag, I-tag para cada etiqueta, más O
                )
                self.entity_extractor.to(DEVICE)
                logger.info("Modelo base cargado como respaldo")
    
    def preprocess_document(self, image):
        """Preprocesa una imagen para el modelo LayoutLMv3"""
        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Extraer texto con pytesseract
        ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Extraer palabras y coordenadas
        words = []
        boxes = []
        for i in range(len(ocr_result["text"])):
            word = ocr_result["text"][i].strip()
            if word:
                words.append(word)
                x = ocr_result["left"][i]
                y = ocr_result["top"][i]
                w = ocr_result["width"][i]
                h = ocr_result["height"][i]
                boxes.append([x, y, x + w, y + h])
        
        # Codificar para el modelo
        encoding = self.processor(
            image, 
            words, 
            boxes=boxes, 
            truncation=True, 
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Mover a GPU si está disponible
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(DEVICE)
        
        return encoding, words, boxes
    
    def is_valid_factura(self, image):
        """Determina si una imagen es una factura válida"""
        self.load_doc_classifier()
        
        encoding, _, _ = self.preprocess_document(image)
        
        # Clasificar el documento
        with torch.no_grad():
            outputs = self.doc_classifier(**encoding)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][prediction].item()
        
        return prediction == 1, confidence  # 1 = factura válida, 0 = no válida
    
    def extract_entities(self, image):
        """Extrae entidades de una factura usando el modelo fine-tuneado"""
        self.load_entity_extractor()
        
        encoding, words, boxes = self.preprocess_document(image)
        
        # Extraer entidades
        with torch.no_grad():
            outputs = self.entity_extractor(**encoding)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        # Procesar resultados
        results = {}
        current_entity = None
        current_text = []
        
        # Crear mapeo de IDs a etiquetas según tu modelo
        # Esta parte es esencial para interpretar correctamente las predicciones
        label_list = ["O"]  # Comenzar con la etiqueta Outside
        for label_name in self.entity_labels:
            label_list.append(f"B-{label_name}")  # Begin
            label_list.append(f"I-{label_name}")  # Inside
        
        id2label = {i: label for i, label in enumerate(label_list)}
        
        for idx, (word, prediction_id) in enumerate(zip(words, predictions)):
            if prediction_id >= len(id2label):
                logger.warning(f"Predicción con ID {prediction_id} fuera de rango (max {len(id2label)-1})")
                continue
                
            label = id2label[prediction_id]
            
            # Si es "O" (outside)
            if label == "O":
                if current_entity and current_text:
                    entity_text = " ".join(current_text)
                    results[current_entity] = entity_text
                    current_entity = None
                    current_text = []
            # Si es un B- (begin)
            elif label.startswith("B-"):
                if current_entity and current_text:
                    entity_text = " ".join(current_text)
                    results[current_entity] = entity_text
                    current_text = []
                current_entity = label[2:]  # Quitar el "B-"
                current_text.append(word)
            # Si es un I- (inside)
            elif label.startswith("I-"):
                entity_name = label[2:]  # Quitar el "I-"
                if entity_name == current_entity:  # Solo agregar si estamos en la misma entidad
                    current_text.append(word)
        
        # No olvidar la última entidad
        if current_entity and current_text:
            entity_text = " ".join(current_text)
            results[current_entity] = entity_text
        
        return results
    
    def verificar_predicciones(self, entities):
        """Verifica la validez de las predicciones del modelo"""
        confianza = 0
        # Verificar entidades críticas
        entidades_criticas = ["CUIT_PRESTADOR", "PUNTO_VENTA", "NUMERO_FACTURA"]
        for entidad in entidades_criticas:
            if entidad in entities:
                # Verificaciones específicas por tipo de entidad
                if entidad == "CUIT_PRESTADOR" and len(entities[entidad].replace("-", "").replace(" ", "")) == 11:
                    confianza += 0.33
                elif entidad == "PUNTO_VENTA" and any(c.isdigit() for c in entities[entidad]):
                    confianza += 0.33
                elif entidad == "NUMERO_FACTURA" and any(c.isdigit() for c in entities[entidad]):
                    confianza += 0.33
        
        # Devolver nivel de confianza y mensaje
        if confianza >= 0.9:
            return True, "Predicciones de alta confianza"
        elif confianza >= 0.66:
            return True, "Predicciones aceptables"
        else:
            return False, f"Predicciones de baja confianza ({confianza:.2f})"

# Crear una instancia global del procesador
factura_processor = None

def inicializar_modelo():
    """Inicializa el modelo de IA"""
    global factura_processor
    check_gpu()
    factura_processor = FacturaProcessor()

def registrar_resultados_modelo(entities, metodo_final, texto_factura):
    """Registra los resultados del modelo para análisis posterior"""
    try:
        # Crear archivo si no existe
        if not os.path.exists('resultados_modelo.csv'):
            with open('resultados_modelo.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Método_Final', 'Entidades_Encontradas', 'Texto_Factura'])
        
        # Registrar resultado
        with open('resultados_modelo.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                metodo_final,
                str(entities.keys()),
                texto_factura[:500] if texto_factura else ""  # Primeros 500 caracteres
            ])
    except Exception as e:
        logger.error(f"Error al registrar resultados del modelo: {e}")

# ========== FUNCIONES DE AUTENTICACIÓN ==========
def autenticar():
    """Autenticación con Google Drive API."""
    creds = None
    
    # Verificar si existe token.json
    if os.path.exists('token.json'):
        try:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            
            # Verificar si las credenciales son válidas
            if not creds.valid:
                if creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception as e:
                        logger.warning(f"Error al refrescar el token: {e}")
                        # Si hay error al refrescar, eliminamos el token y forzamos reautenticación
                        os.remove('token.json')
                        creds = None
                else:
                    # Si las credenciales no son válidas y no podemos refrescarlas, eliminar
                    os.remove('token.json')
                    creds = None
        except Exception as e:
            logger.warning(f"Error con el archivo token.json: {e}")
            # Si hay algún error con el archivo de token, lo eliminamos
            os.remove('token.json')
            creds = None
    
    # Si no hay credenciales válidas, iniciar flujo de autenticación
    if not creds:
        try:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
            # Guardar el token para la próxima vez
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
            
            logger.info("Nuevas credenciales generadas correctamente")
        except Exception as e:
            logger.error(f"Error en el flujo de autenticación: {e}")
            raise Exception(f"No se pudo autenticar con Google Drive: {e}")
    
    return build('drive', 'v3', credentials=creds)

# ========== FUNCIONES PARA RENOMBRAR FACTURAS ==========
def limpiar_numero(num_str):
    """Elimina ceros a la izquierda de un número."""
    try:
        # Primero, elimina cualquier caracter que no sea número
        num_str = ''.join(c for c in str(num_str) if c.isdigit())
        # Luego convierte a entero para eliminar ceros a la izquierda y vuelve a string
        return str(int(num_str)) if num_str else "0"
    except:
        return str(num_str) if num_str else "0"

def extraer_cuits(texto):
    """Extrae todos los CUITs/CUILs presentes en el texto."""
    cuits = []
    # Buscar patrones comunes de CUIT/CUIL (con o sin guiones)
    patrones = [
        r"\bCUIT:?\s*(\d{2}-\d{8}-\d)\b",  # CUIT: XX-XXXXXXXX-X
        r"\bCUIT:?\s*(\d{11})\b",          # CUIT: XXXXXXXXXXX
        r"\bCUIL:?\s*(\d{2}-\d{8}-\d)\b",  # CUIL: XX-XXXXXXXX-X
        r"\bCUIL:?\s*(\d{11})\b",          # CUIL: XXXXXXXXXXX
        r"\b(\d{2}-\d{8}-\d)\b",           # XX-XXXXXXXX-X
        r"\b(\d{11})\b"                   # XXXXXXXXXXX (11 dígitos juntos)
    ]
    
    for patron in patrones:
        matches = re.finditer(patron, texto, re.I)
        for match in matches:
            cuit = match.group(1).replace("-", "")
            if cuit not in cuits and len(cuit) == 11:
                cuits.append(cuit)
    
    return cuits

def extraer_cuit(texto):
    """Extrae el CUIT del prestador del texto de la factura."""
    match = re.search(r'\b(\d{2}-?\d{8}-?\d{1})\b', texto)
    if match:
        return match.group(1).replace("-", "")
    return None

def extraer_cuil_afiliado(texto):
    """
    Extrae el CUIL del afiliado/beneficiario del texto de la factura.
    Este es diferente del CUIT del prestador.
    """
    # Buscar patrones específicos que identifiquen al afiliado
    patrones = [
        r"(?:Paciente|Beneficiario|Afiliado)[\s:]+.*?(?:DNI|CUIL)[:\s]+.*?(\d{2}[-\s]?\d{8}[-\s]?\d{1})",
        r"(?:Paciente|Beneficiario|Afiliado)[:\s]+.*?(\d{2}[-\s]?\d{8}[-\s]?\d{1})",
        r"Datos del (?:paciente|beneficiario|afiliado)[:\s]+.*?CUIL[:\s]+(\d{2}[-\s]?\d{8}[-\s]?\d{1})",
        r"Datos del (?:paciente|beneficiario|afiliado)[:\s]+.*?CUIL[:\s]+(\d{11})",
        r"(?:CUIL|DNI) del (?:paciente|beneficiario|afiliado)[:\s]+(\d{2}[-\s]?\d{8}[-\s]?\d{1})",
        r"(?:CUIL|DNI) del (?:paciente|beneficiario|afiliado)[:\s]+(\d{11})"
    ]
    
    for patron in patrones:
        match = re.search(patron, texto, re.IGNORECASE | re.DOTALL)
        if match:
            cuil = match.group(1).replace("-", "").replace(" ", "")
            if len(cuil) == 11:
                return cuil
    
    # Si no encuentra con los patrones específicos, buscar todos los CUIT/CUIL
    cuits = extraer_cuits(texto)
    if len(cuits) > 1:  # Si hay más de un CUIT/CUIL
        return cuits[1]  # Devolver el segundo (asumiendo que el primero es el prestador)
    
    return None

def extraer_nombre_afiliado(texto):
    """Extrae el nombre del afiliado del texto de la factura."""
    patrones = [
        r"(?:Paciente|Beneficiario|Afiliado)[:\s]+([A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s]+)(?:DNI|CUIL)",
        r"Datos del (?:paciente|beneficiario|afiliado)[:\s]+([A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s]+)(?:DNI|CUIL)",
        r"Nombre del (?:paciente|beneficiario|afiliado)[:\s]+([A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s]+)"
    ]
    
    for patron in patrones:
        match = re.search(patron, texto, re.IGNORECASE | re.DOTALL)
        if match:
            nombre = match.group(1).strip()
            if len(nombre) > 3:  # Verificar que sea un nombre válido
                return nombre
    
    return None

def extraer_tipo_y_letra(texto):
    """Extrae el tipo de comprobante (FACTURA/RECIBO) y letra (B/C)."""
    # 1) Tipo
    tipo_match = re.search(r'(FACTURA|RECIBO)', texto, re.IGNORECASE)
    if not tipo_match:
        return None, None
    tipo = tipo_match.group(1).upper()

    # 2) Letra:
    # A) ([BC])COD.
    letra_cod = re.search(r'([BC])\s*COD\.?\s*\d*', texto, re.IGNORECASE)
    if letra_cod:
        return tipo, letra_cod.group(1).upper()

    # B) Letra suelta
    letra_suelta = re.search(r'(^|\n)\s*([BC])\s*(\n|$)', texto, re.IGNORECASE)
    if letra_suelta:
        return tipo, letra_suelta.group(2).upper()

    # C) "B(\d\d+)" => B006, C011, etc. => si ves "C011" interpretamos letra = C
    letra_numerica = re.search(r'\b([BC])\d{2,4}\b', texto, re.IGNORECASE)
    if letra_numerica:
        return tipo, letra_numerica.group(1).upper()

    return tipo, None

def tipo_y_letra_a_codigo(tipo, letra):
    """Convierte tipo y letra a código de comprobante AFIP."""
    if not tipo or not letra:
        return None
    
    key = (letra, tipo)
    if key in CODIGO_CBTE:
        return int(CODIGO_CBTE[key])
    
    # Si no está en el mapa, intentar búsqueda flexible
    for (l, t), codigo in CODIGO_CBTE.items():
        if l == letra and t.startswith(tipo[:3]):
            return int(codigo)
    
    return None

def extraer_pv_nro(texto):
    """
    Extrae punto de venta y número de comprobante.
    """
    # Plan A
    patron_a = re.search(r'Punto\s*de\s*Venta:\s*.*Comp\.?\s*Nro:\s*0*(\d+)\s+0*(\d+)', texto, re.DOTALL)
    if patron_a:
        return limpiar_numero(patron_a.group(1)), limpiar_numero(patron_a.group(2))

    # Plan B: 'Nro 00004-00003575'
    patron_b = re.search(r'Nro\s+0*(\d+)-0*(\d+)', texto)
    if patron_b:
        return limpiar_numero(patron_b.group(1)), limpiar_numero(patron_b.group(2))

    # Plan C: '0002-00027842' a secas
    patron_c = re.search(r'\b0*(\d+)-0*(\d+)\b', texto)
    if patron_c:
        return limpiar_numero(patron_c.group(1)), limpiar_numero(patron_c.group(2))

    # Plan D: (FAC-)?B-0003-00002475 con espacios
    patron_d = re.search(r'(FAC\-)?([BC])\s*-\s*0*(\d+)\s*-\s*0*(\d+)', texto, re.IGNORECASE)
    if patron_d:
        return limpiar_numero(patron_d.group(3)), limpiar_numero(patron_d.group(4))

    # Plan E: (FAC-)?B-0003-00002475 sin espacios
    #  => "B-0003-00002475", "FAC-B-0003-00002475"
    patron_e = re.search(r'(FAC\-)?([BC])\-0*(\d+)\-0*(\d+)', texto, re.IGNORECASE)
    if patron_e:
        return limpiar_numero(patron_e.group(3)), limpiar_numero(patron_e.group(4))

    return None, None

def extraer_datos_flexible(texto):
    """Extrae todos los datos necesarios para renombrar el archivo."""
    # 1) CUIT del prestador
    cuit = extraer_cuit(texto)
    if not cuit:
        return None

    # 2) CUIL del afiliado y nombre (para ayudar en la correspondencia)
    cuil_afiliado = extraer_cuil_afiliado(texto)
    nombre_afiliado = extraer_nombre_afiliado(texto)

    # 3) Tipo y Letra
    tipo, letra = extraer_tipo_y_letra(texto)
    codigo = tipo_y_letra_a_codigo(tipo, letra)

    # 4) PV y Nro
    pv, nro = extraer_pv_nro(texto)

    # 5) Fallback si la letra no salió
    if (not codigo) and letra is None:
        # Plan D/E: "B-0003-00002475"
        plan_de_letra = re.search(r'(FAC\-)?([BC])\-0*(\d+)\-0*(\d+)', texto, re.IGNORECASE)
        if plan_de_letra:
            let = plan_de_letra.group(2).upper()
            if tipo:
                codigo = tipo_y_letra_a_codigo(tipo, let)

    if not (cuit and codigo and pv and nro):
        return None

    return {
        'cuit': cuit,                    # CUIT del prestador
        'codigo': codigo,                # Código de comprobante
        'pv': pv,                        # Punto de venta
        'nro': nro,                      # Número de comprobante
        'cuil_afiliado': cuil_afiliado,  # CUIL del afiliado (si se encontró)
        'nombre_afiliado': nombre_afiliado,  # Nombre del afiliado (si se encontró)
        'nombre_archivo': f"{cuit}_{codigo}_{pv}_{nro}.pdf"
    }

def leer_pagina(pdf_bytes, pagina):
    """Lee una página de PDF y extrae su texto."""
    try:
        reader = PdfReader(pdf_bytes)
        if pagina >= len(reader.pages):
            return ""
        return reader.pages[pagina].extract_text() or ""
    except:
        return ""

def leer_pagina_ocr(pdf_bytes, pagina):
    """Aplica OCR a una página de PDF para extraer texto."""
    try:
        images = convert_from_bytes(pdf_bytes.getvalue(), first_page=pagina+1, last_page=pagina+1)
        return pytesseract.image_to_string(images[0])
    except:
        return ""

def extraer_desde_dos_paginas(fh, use_ai=True):
   """Extrae datos desde las dos primeras páginas del PDF usando OCR y IA."""
   texto_1 = leer_pagina(fh, 0)
   texto_2 = leer_pagina(fh, 1)
   
   logger.info("\n--- TEXTO PÁGINA 1 ---\n" + texto_1[:1000])
   logger.info("\n--- TEXTO PÁGINA 2 ---\n" + texto_2[:1000])
   
   # OCR si ambas casi vacías
   if len((texto_1 + texto_2).strip()) < 20:
       texto_1 = leer_pagina_ocr(fh, 0)
       texto_2 = leer_pagina_ocr(fh, 1)
       logger.info("\n--- TEXTO PÁGINA 1 (OCR) ---\n" + texto_1[:1000])
       logger.info("\n--- TEXTO PÁGINA 2 (OCR) ---\n" + texto_2[:1000])
   
   fh.seek(0)
   
   # Intentar con IA PRIMERO si está habilitada (cambio importante)
   if use_ai and factura_processor:
       logger.info("Intentando extraer con IA primero...")
       try:
           # Convertir la primera página a imagen para IA
           fh.seek(0)
           images = convert_from_bytes(fh.getvalue(), first_page=1, last_page=1)
           if images:
               image = images[0]
               
               # Verificar si es una factura válida
               is_valid, confidence = factura_processor.is_valid_factura(image)
               logger.info(f"IA: ¿Es factura válida? {is_valid} (confianza: {confidence:.2f})")
               
               # Extraer entidades con IA independientemente de si detecta que es una factura válida
               # (esto nos permite capturar más casos)
               entities = factura_processor.extract_entities(image)
               logger.info(f"Entidades extraídas por IA: {entities}")
               
               # Verificar confianza de predicciones
               es_confiable, mensaje = factura_processor.verificar_predicciones(entities)
               logger.info(f"Evaluación del modelo: {mensaje}")
               
               # Construir datos desde entidades si hay suficientes entidades críticas
               if es_confiable and 'CUIT_PRESTADOR' in entities and 'PUNTO_VENTA' in entities and 'NUMERO_FACTURA' in entities:
                   cuit = entities['CUIT_PRESTADOR'].replace("-", "").replace(" ", "")
                   
                   # Determinar código de comprobante - más flexible
                   tipo = 'FACTURA'  # Por defecto
                   letra = 'C'  # Por defecto
                   
                   if 'TIPO_FACTURA' in entities:
                       tipo = entities['TIPO_FACTURA']
                   if 'LETRA_FACTURA' in entities:
                       letra = entities['LETRA_FACTURA']
                   
                   # Intentar obtener el código
                   codigo = tipo_y_letra_a_codigo(tipo, letra.upper())
                   
                   # Si no se pudo determinar el código, usar valor por defecto (5 = Factura C)
                   if not codigo:
                       codigo = 5
                       logger.info(f"IA: No se pudo determinar código, usando valor por defecto {codigo}")
                   
                   pv = limpiar_numero(entities['PUNTO_VENTA'])
                   nro = limpiar_numero(entities['NUMERO_FACTURA'])
                   
                   # Crear datos con información de IA
                   datos_ia = {
                       'cuit': cuit,
                       'codigo': codigo,
                       'pv': pv,
                       'nro': nro,
                       'nombre_archivo': f"{cuit}_{codigo}_{pv}_{nro}.pdf",
                       'texto_completo': texto_1 + "\n" + texto_2,  # Mantener texto original
                   }
                   
                   # Agregar datos adicionales si existen
                   if 'CUIT_AFILIADO' in entities:
                       datos_ia['cuil_afiliado'] = entities['CUIT_AFILIADO'].replace("-", "").replace(" ", "")
                   if 'NOMBRE_AFILIADO' in entities:
                       datos_ia['nombre_afiliado'] = entities['NOMBRE_AFILIADO']
                   if 'DNI_AFILIADO' in entities:
                       datos_ia['dni_afiliado'] = entities['DNI_AFILIADO'].replace(" ", "")
                   
                   # Registrar éxito del modelo
                   registrar_resultados_modelo(entities, "IA", texto_1 + "\n" + texto_2)
                   
                   logger.info(f"IA extrajo datos válidos: {datos_ia['nombre_archivo']}")
                   return datos_ia, None
               else:
                   logger.warning(f"IA no pudo extraer datos suficientes o confiables: {mensaje}")
       except Exception as e:
           logger.error(f"Error en procesamiento con IA: {e}")
   
   # Si la IA falló o no está habilitada, volver al método tradicional
   logger.info("Usando método tradicional como respaldo...")
   datos1 = extraer_datos_flexible(texto_1)
   datos2 = extraer_datos_flexible(texto_2)
   
   datos_completos = None
   error = None
   
   if datos1 and datos2:
       if datos1['nombre_archivo'] == datos2['nombre_archivo']:
           datos_completos = datos1
           datos_completos['texto_completo'] = texto_1 + "\n" + texto_2
       else:
           error = '❌ Inconsistencia entre página 1 y 2'
   elif datos1:
       datos_completos = datos1
       datos_completos['texto_completo'] = texto_1 + "\n" + texto_2
   elif datos2:
       datos_completos = datos2
       datos_completos['texto_completo'] = texto_1 + "\n" + texto_2
   else:
       error = '❌ No se pudo extraer información de ninguna página'
   
   # Registrar resultado del método tradicional
   if datos_completos:
       metodo = "Tradicional"
       entities_dummy = {k: v for k, v in datos_completos.items() if k not in ['texto_completo', 'nombre_archivo']}
       registrar_resultados_modelo(entities_dummy, metodo, texto_1 + "\n" + texto_2)
   
   return datos_completos, error

def extraer_desde_imagen(image_bytes):
   """Extrae datos desde una imagen."""
   try:
       image = Image.open(io.BytesIO(image_bytes))
       texto = pytesseract.image_to_string(image)
       datos = extraer_datos_flexible(texto)
       if datos:
           datos['texto_completo'] = texto
           return datos, None
           
       # Intentar con IA si hay disponible y el método tradicional falló
       if factura_processor:
           try:
               # Usar modelo para extraer entidades
               entities = factura_processor.extract_entities(image)
               logger.info(f"IA extrajo entidades de imagen: {entities}")
               
               # Verificar confianza de predicciones
               es_confiable, mensaje = factura_processor.verificar_predicciones(entities)
               logger.info(f"Evaluación del modelo en imagen: {mensaje}")
               
               if es_confiable:
                   # Construir datos desde entidades
                   if 'CUIT_PRESTADOR' in entities and 'PUNTO_VENTA' in entities and 'NUMERO_FACTURA' in entities:
                       cuit = entities['CUIT_PRESTADOR'].replace("-", "").replace(" ", "")
                       
                       # Determinar código de comprobante
                       tipo = 'FACTURA' if 'TIPO_FACTURA' in entities else 'FACTURA'  # Por defecto
                       letra = entities.get('LETRA_FACTURA', 'C')  # C por defecto
                       
                       codigo = tipo_y_letra_a_codigo(tipo, letra.upper())
                       if not codigo:
                           codigo = 5  # Valor por defecto (Factura C)
                       
                       pv = limpiar_numero(entities['PUNTO_VENTA'])
                       nro = limpiar_numero(entities['NUMERO_FACTURA'])
                       
                       # Crear datos con información de IA
                       datos_ia = {
                           'cuit': cuit,
                           'codigo': codigo,
                           'pv': pv,
                           'nro': nro,
                           'nombre_archivo': f"{cuit}_{codigo}_{pv}_{nro}.pdf",
                           'texto_completo': texto,  # Mantener texto original
                       }
                       
                       # Agregar datos adicionales si existen
                       if 'CUIT_AFILIADO' in entities:
                           datos_ia['cuil_afiliado'] = entities['CUIT_AFILIADO'].replace("-", "").replace(" ", "")
                       if 'NOMBRE_AFILIADO' in entities:
                           datos_ia['nombre_afiliado'] = entities['NOMBRE_AFILIADO']
                       
                       # Registrar éxito del modelo
                       registrar_resultados_modelo(entities, "IA_Imagen", texto)
                       
                       logger.info(f"IA extrajo datos válidos de la imagen: {datos_ia['nombre_archivo']}")
                       return datos_ia, None
           except Exception as e:
               logger.error(f"Error en procesamiento de imagen con IA: {e}")
       
       return None, '❌ No se pudo extraer información desde imagen'
   except Exception as e:
       return None, f'OCR Error en imagen: {e}'

def extraer_datos_desde_nombre(nombre):
   """Extrae datos desde el nombre del archivo si ya está renombrado."""
   match = re.match(r'^(\d{11})_(\d+)_(\d+)_(\d+)\.pdf$', nombre)
   if match:
       cuit = match.group(1)
       codigo = int(match.group(2))
       pv = match.group(3)
       nro = match.group(4)
       return {
           'cuit': cuit,
           'codigo': codigo,
           'pv': pv,
           'nro': nro,
           'nombre_archivo': nombre
       }
   return None

# ========== FUNCIONES PARA GENERACIÓN DE TXT ==========
def formatear_fecha(fecha_str):
   """Convierte una fecha a formato DD/MM/AAAA."""
   if not fecha_str:
       return ""
       
   fecha_str = str(fecha_str).strip()
   
   try:
       # Si es una fecha en formato ISO (YYYY-MM-DD)
       import re
       from datetime import datetime
       
       # Patrón ISO (YYYY-MM-DD) con o sin hora
       iso_pattern = re.match(r'(\d{4})-(\d{2})-(\d{2})(\s.*)?', fecha_str)
       if iso_pattern:
           year, month, day = iso_pattern.groups()[:3]
           return f"{day}/{month}/{year}"
           
       # Patrón con barras pero otro formato (YYYY/MM/DD)
       slash_pattern = re.match(r'(\d{4})/(\d{2})/(\d{2})(\s.*)?', fecha_str)
       if slash_pattern:
           year, month, day = slash_pattern.groups()[:3]
           return f"{day}/{month}/{year}"
           
       # Si ya está en formato DD/MM/AAAA, verificar y mantener
       ddmmyyyy_pattern = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})(\s.*)?', fecha_str)
       if ddmmyyyy_pattern:
           day, month, year = ddmmyyyy_pattern.groups()[:3]
           # Asegurar que tengan dos dígitos
           day = day.zfill(2)
           month = month.zfill(2)
           return f"{day}/{month}/{year}"
           
       # Intentar parsear con datetime si no coincide con patrones anteriores
       for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y'):
           try:
               dt = datetime.strptime(fecha_str, fmt)
               return dt.strftime('%d/%m/%Y')
           except ValueError:
               continue
               
       # Si llega aquí, mantener el formato original y advertir
       logger.warning(f"No se pudo formatear la fecha: {fecha_str}")
       return fecha_str
       
   except Exception as e:
       logger.error(f"Error al formatear fecha '{fecha_str}': {e}")
       return fecha_str

def extraer_fecha(texto):
   """Extrae la fecha de emisión del documento."""
   # Buscar fecha explícita de emisión
   fecha_match = re.search(r"Fecha\s+de\s+Emisión:?\s*(\d{2}/\d{2}/\d{4})", texto, re.I)
   if fecha_match:
       return fecha_match.group(1)
   
   # Si no hay fecha explícita, buscar todas las fechas
   fechas = re.findall(r"\d{2}/\d{2}/\d{4}", texto)
   if len(fechas) >= 4:
       # En muchas facturas, la 4ª fecha es la fecha de emisión
       return fechas[3]
   elif fechas:
       # Si no hay 4 fechas, usar la primera
       return fechas[0]
   
   # Si no hay fechas con formato estándar, intentar otros formatos
   alt_fechas = re.findall(r"\d{2}-\d{2}-\d{4}", texto)
   if alt_fechas:
       return alt_fechas[0].replace("-", "/")
   
   # Patrones adicionales para fechas
   patrones_fecha = [
       r"Fecha[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
       r"(\d{1,2}[-/]\d{1,2}[-/](?:20|19)\d{2})",  # Fechas con año de 4 dígitos
       r"Emisión[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
   ]
   
   for patron in patrones_fecha:
       match = re.search(patron, texto, re.IGNORECASE)
       if match:
           fecha_str = match.group(1)
           # Normalizar separadores
           fecha_str = fecha_str.replace('-', '/')
           # Verificar formato
           partes = fecha_str.split('/')
           if len(partes) == 3:
               # Si el año tiene 2 dígitos, convertir a 4 dígitos
               if len(partes[2]) == 2:
                   partes[2] = '20' + partes[2]
               # Asegurar que día y mes tengan 2 dígitos
               partes[0] = partes[0].zfill(2)
               partes[1] = partes[1].zfill(2)
               return '/'.join(partes)
   
   # Si no hay fechas, usar la fecha actual
   return datetime.now().strftime("%d/%m/%Y")

def extraer_cae(texto):
   """Extrae el CAE/CAI y el tipo de emisión (E/I)."""
   # Buscar CAE explícito
   cae_match = re.search(r"CAE\s*N°:?\s*(\d{14})", texto, re.I)
   if cae_match:
       return cae_match.group(1), "E"
   
   # Buscar al final del documento (común en muchas facturas)
   cae_final = re.search(r"Pág\.\s*1/1\s+(\d{14})", texto)
   if cae_final:
       return cae_final.group(1), "E"
   
   # Buscar cualquier número de 14 dígitos cerca de "CAE"
   cae_cercano = re.search(r"CAE.*?(\d{14})", texto, re.DOTALL | re.I)
   if cae_cercano:
       return cae_cercano.group(1), "E"
   
   # Buscar CAI (Código de Autorización de Impresión)
   cai_match = re.search(r"CAI\s*N°:?\s*(\d{14})", texto, re.I)
   if cai_match:
       return cai_match.group(1), "I"
   
   # Patrones adicionales para CAE
   patrones_cae = [
       r"C\.?A\.?E\.?[:\s#]+(\d{14})",
       r"Nro\.?\s+de\s+Comp\.?\s+Autorizado[:\s]+.*?(\d{14})",
       r"Autorización[:\s]+.*?(\d{14})"
   ]
   
   for patron in patrones_cae:
       match = re.search(patron, texto, re.IGNORECASE)
       if match:
           return match.group(1), "E"
   
   # Última opción: cualquier número de 14 dígitos en el documento
   cualquier_14 = re.search(r"\b(\d{14})\b", texto)
   if cualquier_14:
       return cualquier_14.group(1), "E"  # Asumimos E por ser más común
   
   return None, None

def extraer_importe(texto):
   """Extrae el importe total del documento."""
   # Patrones de búsqueda del importe
   patrones = [
       r"Importe Total:?\s*\$?\s*([\d.,]+)",
       r"TOTAL:?\s*\$?\s*([\d.,]+)",
       r"Total:?\s*\$?\s*([\d.,]+)",
       r"IMPORTE TOTAL:?\s*\$?\s*([\d.,]+)",
   ]
   
   # Buscar importe con etiqueta
   for patron in patrones:
       m = re.search(patron, texto, re.I)
       if m:
           # Limpiar formato de números
           importe = m.group(1).replace(".", "").replace(",", "")
           return importe
   
   # Si no hay etiqueta, buscar importes numéricos con formato
   importes = re.findall(r"(\d{3,6}(?:\.\d{3})*(?:,\d{2}))", texto)
   if importes:
       # El último importe suele ser el total
       ultimo_importe = importes[-1]
       return ultimo_importe.replace(".", "").replace(",", "")
   
   return None

def extraer_periodo(texto):
   """Extrae el período facturado en formato MMAAAA."""
   # Buscar "Período Facturado Desde: dd/mm/yyyy Hasta: dd/mm/yyyy"
   periodo_match = re.search(r"Período\s*Facturado\s*Desde:?\s*\d{1,2}/(\d{1,2})/(\d{4})", texto, re.I)
   if periodo_match:
       mes = periodo_match.group(1).zfill(2)  # Asegurar 2 dígitos
       anio = periodo_match.group(2)
       return f"{mes}{anio}"
   
   # Buscar menciones al "mes de XXX de YYYY"
   mes_match = re.search(r"mes\s+de\s+(\w+)\s+de\s+(\d{4})", texto, re.I)
   if mes_match:
       mes_texto = mes_match.group(1).lower()
       año = mes_match.group(2)
       meses = {
           "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
           "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
           "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"
       }
       if mes_texto in meses:
           return f"{meses[mes_texto]}{año}"
   
   # Buscar "Período: MM/YYYY" o similar
   patron_periodo = re.search(r"Período[:\s]+(\d{1,2})/(\d{4})", texto, re.IGNORECASE)
   if patron_periodo:
       mes = patron_periodo.group(1).zfill(2)
       anio = patron_periodo.group(2)
       return f"{mes}{anio}"
   
   # Si no encontramos nada, usar el mes actual
   ahora = datetime.now()
   return f"{ahora.month:02d}{ahora.year}"

def map_actividad(texto):
   """Determina el código de actividad y la bandera de dependencia."""
   texto_lower = texto.lower()
   
   # Transporte
   if "transporte" in texto_lower or "traslado" in texto_lower or re.search(r"\bkm\b", texto_lower):
       dep_flag = "S" if any(term in texto_lower for term in ["dependencia", "discapacidad", "discapac"]) else "N"
       return "096", dep_flag
   
   # Prestaciones profesionales (psicología, etc.)
   if any(term in texto_lower for term in [
       "psicología", "psicologia", "psicólogo", "psicologo",
       "musicoterapia", "musicoterapeuta",
       "kinesiología", "kinesiologia", "kinesiólogo", "kinesiologo",
       "fonoaudiología", "fonoaudiologia", "fonoaudiólogo", "fonoaudiologo",
       "psicopedagogía", "psicopedagogia", "psicopedagogo"
   ]):
       return "091", "N"
   
   # Estimulación temprana
   if "estimulación temprana" in texto_lower or "estimulacion temprana" in texto_lower:
       return "085", "N"
   
   # Apoyo a la integración escolar
   if any(term in texto_lower for term in [
       "módulo de apoyo", "modulo de apoyo", 
       "apoyo a la integración", "apoyo a la integracion",
       "maestra integradora", "maestro integrador"
   ]):
       return "089", "N"
   
   # Actividades terapéuticas
   if "honorarios profesionales" in texto_lower or "sesiones" in texto_lower or "terapia" in texto_lower:
       return "090", "N"
   
   # Por defecto, usar código genérico
   dep_flag = "S" if any(term in texto_lower for term in ["dependencia", "discapacidad", "discapac"]) else "N"
   return "090", dep_flag

def cantidad_por_actividad(cod):
   """Devuelve la cantidad predeterminada según el código de actividad."""
   if cod == "096":  # Transporte
       return "001500"
   if cod in {"090", "091"}:  # Terapias y profesionales
       return "000004"
   # Actividades mensuales
   return "000001"

def extraer_datos_adicionales(datos, use_ai=True):
   """Extrae datos adicionales para la generación del TXT, priorizando IA."""
   texto = datos.get('texto_completo', '')
   if not texto:
       return None
   
   # Datos ya existentes del renombrado
   cuit = datos.get('cuit')
   codigo_cbte = str(datos.get('codigo')).zfill(2)
   pv = datos.get('pv')
   nro = datos.get('nro')
   
   # Variables para almacenar resultados
   fecha_cbte = None
   nro_cae = None
   tipo_emision = None
   importe = None
   periodo = None
   actividad = None
   dep_flag = None
   
   # SIEMPRE intentar con IA primero si está disponible
   if use_ai and factura_processor:
       try:
           # Obtener el file_id si existe
           file_id = datos.get('file_id')
           if file_id:
               logger.info(f"Extrayendo datos adicionales con IA para {datos.get('nombre_archivo')}")
               
               # Obtener el servicio de Drive (debe estar disponible en el contexto)
               service = autenticar()
               
               # Descargar el PDF
               request = service.files().get_media(fileId=file_id)
               fh = io.BytesIO()
               downloader = MediaIoBaseDownload(fh, request)
               done = False
               while not done:
                   status, done = downloader.next_chunk()
               fh.seek(0)
               
               # Convertir a imagen
               images = convert_from_bytes(fh.getvalue(), first_page=1, last_page=1)
               if images:
                   image = images[0]
                   
                   # Extraer entidades con IA
                   entities = factura_processor.extract_entities(image)
                   logger.info(f"IA extrajo entidades adicionales: {entities}")
                   
                   # Extraer datos de las entidades
                   if 'FECHA_EMISION' in entities:
                       fecha_cbte = formatear_fecha(entities['FECHA_EMISION'])
                       logger.info(f"IA encontró fecha: {fecha_cbte}")
                   
                   if 'CAE' in entities:
                       nro_cae = entities['CAE']
                       tipo_emision = "E"  # Si hay CAE, es electrónica
                       logger.info(f"IA encontró CAE: {nro_cae}")
                   
                   if 'IMPORTE' in entities:
                       importe_str = entities['IMPORTE']
                       # Limpiar el importe
                       importe_str = ''.join(c for c in importe_str if c.isdigit() or c in [',', '.'])
                       importe = importe_str.replace(",", "").replace(".", "")
                       logger.info(f"IA encontró importe: {importe}")
                   
                   if 'PERIODO' in entities:
                       periodo_str = entities['PERIODO']
                       # Intentar extraer mes y año
                       periodo_match = re.search(r"(\d{1,2})[/-](\d{4})", periodo_str)
                       if periodo_match:
                           mes = periodo_match.group(1).zfill(2)
                           anio = periodo_match.group(2)
                           periodo = f"{mes}{anio}"
                           logger.info(f"IA encontró periodo: {periodo}")
                       
                   if 'ACTIVIDAD' in entities:
                       # Asignar actividad según palabras clave en el texto extraído
                       actividad_texto = entities['ACTIVIDAD'].lower()
                       
                       if "transport" in actividad_texto or "traslad" in actividad_texto:
                           actividad = "096"
                           dep_flag = "S" if "discapac" in actividad_texto else "N"
                       elif any(term in actividad_texto for term in 
                              ["psicolog", "terapeut", "kinesio", "fonoaudio"]):
                           actividad = "091"
                           dep_flag = "N"
                       elif "estimul" in actividad_texto:
                           actividad = "085" 
                           dep_flag = "N"
                       elif "apoyo" in actividad_texto or "integr" in actividad_texto:
                           actividad = "089"
                           dep_flag = "N"
                       else:
                           actividad = "090"
                           dep_flag = "N"
                       
                       logger.info(f"IA determinó actividad: {actividad}, dependencia: {dep_flag}")
       except Exception as e:
           logger.error(f"Error al usar IA para datos adicionales: {e}")
   
   # Usar métodos tradicionales como respaldo para los datos que no se pudieron extraer con IA
   if not fecha_cbte:
       fecha_cbte = extraer_fecha(texto)
       logger.info(f"Método tradicional encontró fecha: {fecha_cbte}")
   
   if not nro_cae:
       nro_cae, tipo_emision_trad = extraer_cae(texto)
       if nro_cae:
           tipo_emision = tipo_emision_trad
           logger.info(f"Método tradicional encontró CAE: {nro_cae}")
   
   if not importe:
       importe = extraer_importe(texto)
       logger.info(f"Método tradicional encontró importe: {importe}")
   
   if not periodo:
       periodo = extraer_periodo(texto)
       logger.info(f"Método tradicional encontró periodo: {periodo}")
   
   if not actividad:
       actividad, dep_flag = map_actividad(texto)
       logger.info(f"Método tradicional determinó actividad: {actividad}, dependencia: {dep_flag}")
   
   cantidad = cantidad_por_actividad(actividad)
   
   # Devolver datos completos para TXT
   return {
       'cuit_pre': cuit,
       'codigo_cbte': codigo_cbte,
       'pv': pv,
       'nro': nro,
       'fecha_cbte': fecha_cbte,
       'tipo_emision': tipo_emision or "E",
       'nro_cae': nro_cae or "",
       'importe': importe or "0",
       'periodo': periodo,
       'actividad': actividad,
       'cantidad': cantidad,
       'dep': dep_flag,
       'cuil_afiliado': datos.get('cuil_afiliado'),
       'nombre_afiliado': datos.get('nombre_afiliado'),
       'texto_factura': texto.lower()  # Guardar el texto completo en minúsculas para búsquedas
   }

def verificar_formato_txt(nombre_txt):
   """Verifica que el formato del archivo TXT sea correcto."""
   try:
       with open(nombre_txt, 'r', encoding='utf-8') as f:
           lineas = f.readlines()
       
       errores = []
       for i, linea in enumerate(lineas, 1):
           campos = linea.strip().split('|')
           
           # Verificar cantidad de campos
           if len(campos) != 19:
               errores.append(f"Línea {i}: Número incorrecto de campos ({len(campos)}, debería ser 19)")
               continue
           
           # Verificar formato de fecha de vencimiento
           try:
               fecha_venc = campos[4]
               patron_fecha = re.match(r'(\d{2})/(\d{2})/(\d{4})', fecha_venc)
               if not patron_fecha:
                   errores.append(f"Línea {i}: Formato de fecha de vencimiento incorrecto: {fecha_venc}")
           except:
               errores.append(f"Línea {i}: Error al verificar fecha de vencimiento")
           
           # Verificar formato de fecha de comprobante
           try:
               fecha_cbte = campos[9]
               patron_fecha = re.match(r'(\d{2})/(\d{2})/(\d{4})', fecha_cbte)
               if not patron_fecha:
                   errores.append(f"Línea {i}: Formato de fecha de comprobante incorrecto: {fecha_cbte}")
           except:
               errores.append(f"Línea {i}: Error al verificar fecha de comprobante")
           
           # Verificar longitud del código de certificado
           if len(campos[3]) != 38:
               errores.append(f"Línea {i}: Longitud del código de certificado incorrecta: {len(campos[3])}")
           
           # Verificar longitud de importe
           if len(campos[13]) != 14 or len(campos[14]) != 14:
               errores.append(f"Línea {i}: Longitud de importe incorrecta: {len(campos[13])}, {len(campos[14])}")
       
       if errores:
           logger.warning("Se encontraron errores en el archivo TXT:")
           for error in errores:
               logger.warning(error)
           return False
       
       logger.info("Verificación del archivo TXT: CORRECTA")
       return True
   
   except Exception as e:
       logger.error(f"Error al verificar archivo TXT: {e}")
       return False

def construir_linea(rnos, fila_excel, info_pdf):
   """Construye una línea del archivo TXT con el formato específico."""
   # Validar que tenemos todos los datos necesarios
   for campo_requerido in ["cuit_pre", "codigo_cbte", "pv", "nro", 
                          "fecha_cbte", "tipo_emision", "nro_cae", 
                          "importe", "periodo", "actividad", 
                          "cantidad", "dep"]:
       if campo_requerido not in info_pdf or not info_pdf[campo_requerido]:
           # Proporcionar valores predeterminados para campos faltantes
           if campo_requerido == "tipo_emision":
               info_pdf[campo_requerido] = "E"
           elif campo_requerido == "nro_cae":
               info_pdf[campo_requerido] = "00000000000000"  # 14 ceros
           elif campo_requerido == "importe":
               info_pdf[campo_requerido] = "0"
           elif campo_requerido == "periodo":
               ahora = datetime.now()
               info_pdf[campo_requerido] = f"{ahora.month:02d}{ahora.year}"
           elif campo_requerido == "dep":
               info_pdf[campo_requerido] = "N"
           else:
               logger.warning(f"Falta campo requerido: {campo_requerido}")
               return None
   
   # Asegurar formato correcto de los datos
   pv_limpio = str(info_pdf["pv"]).zfill(5)  # 5 dígitos
   nro_limpio = str(info_pdf["nro"]).zfill(8)  # 8 dígitos
   
   # Asegurar formato correcto del importe (14 dígitos, sin decimales)
   try:
       importe_str = str(info_pdf["importe"])
       # Eliminar puntos y comas
       importe_str = importe_str.replace(".", "").replace(",", "")
       # Eliminar caracteres no numéricos
       importe_str = ''.join(c for c in importe_str if c.isdigit())
       importe_limpio = importe_str.zfill(14)  # Asegurar 14 dígitos
   except:
       importe_limpio = "00000000000000"  # Si hay error, usar 14 ceros
   
   # Asegurar que la fecha de vencimiento del certificado tenga el formato correcto DD/MM/AAAA
   fecha_vencimiento = fila_excel["vencimiento_certificado"]
   fecha_vencimiento = formatear_fecha(fecha_vencimiento)
   
   # Verificar longitud del código de certificado
   codigo_cert = fila_excel["codigo_certificado"]
   if len(codigo_cert) > 38:
       codigo_cert = codigo_cert[:38]
   else:
       codigo_cert = codigo_cert.ljust(38, "0")
   
   # Construir línea
   linea = [
       "DS",                            # Constante
       rnos,                            # RNOS de la obra social
       fila_excel["cuil"],              # CUIL del beneficiario
       codigo_cert,                     # Código de certificado (38 caracteres)
       fecha_vencimiento,               # Vencimiento del certificado (corregido)
       info_pdf["periodo"],             # Período facturado (MMAAAA)
       info_pdf["cuit_pre"],            # CUIT del prestador
       info_pdf["codigo_cbte"],         # Código de comprobante
       info_pdf["tipo_emision"],        # Tipo de emisión (E o I)
       info_pdf["fecha_cbte"],          # Fecha del comprobante
       info_pdf["nro_cae"],             # Número de CAE/CAI
       pv_limpio,                       # Punto de venta (5 dígitos)
       nro_limpio,                      # Número de comprobante (8 dígitos)
       importe_limpio,                  # Importe total (14 dígitos)
       importe_limpio,                  # Importe total [duplicado] (14 dígitos)
       info_pdf["actividad"],           # Código de actividad (3 dígitos)
       info_pdf["cantidad"],            # Cantidad (6 dígitos)
       fila_excel["provincia"],         # Código de provincia (2 dígitos)
       info_pdf["dep"],                 # Indicador de dependencia (S/N)
   ]
   
   return "|".join(linea)

def subir_log(service, filepath, folder_id):
   """Sube un archivo de log a Google Drive."""
   file_metadata = {'name': os.path.basename(filepath), 'parents': [folder_id]}
   media = MediaFileUpload(filepath, mimetype='text/csv')
   service.files().create(body=file_metadata, media_body=media, fields='id').execute()

# ========== FUNCIÓN PRINCIPAL PARA RENOMBRAR ==========
def descargar_y_renombrar(service, use_ai=True):
   """Descarga y renombra los PDFs de facturas en Drive."""
   resultados = []
   archivos = service.files().list(
       q=f"'{FOLDER_ID}' in parents and (mimeType='application/pdf' or mimeType contains 'image/')",
       fields="files(id, name, mimeType)",
       pageSize=1000
   ).execute().get('files', [])

   if not archivos:
       logger.info("No se encontraron archivos.")
       return []

   # Eliminar duplicados basados en id del archivo
   archivos_unicos = {}
   for archivo in archivos:
       archivos_unicos[archivo['id']] = archivo
   
   archivos = list(archivos_unicos.values())
   logger.info(f"Se encontraron {len(archivos)} archivos únicos en la carpeta.")

   renombrados = 0
   errores = 0
   archivos_datos = []  # Lista para guardar los datos de cada archivo procesado

   for archivo in archivos:
       file_id = archivo['id']
       nombre_original = archivo['name']
       tipo = archivo['mimeType']

       # Comprobar si ya está en formato final
       matched = re.match(PATTERN_RENAMED, nombre_original)
       if matched:
           logger.info(f"✅ {nombre_original} ya está en formato correcto, extrayendo datos...")
           # Extraer datos del nombre
           datos_del_nombre = extraer_datos_desde_nombre(nombre_original)
           if datos_del_nombre:
               # Descargar para obtener texto completo
               request = service.files().get_media(fileId=file_id)
               fh = io.BytesIO()
               downloader = MediaIoBaseDownload(fh, request)
               done = False
               while not done:
                   status, done = downloader.next_chunk()
               fh.seek(0)
               
               if tipo == 'application/pdf':
                   texto_1 = leer_pagina(fh, 0)
                   texto_2 = leer_pagina(fh, 1)
                   datos_del_nombre['texto_completo'] = texto_1 + "\n" + texto_2
                   
                   # Intentar extraer CUIL del afiliado
                   cuil_afiliado = extraer_cuil_afiliado(datos_del_nombre['texto_completo'])
                   nombre_afiliado = extraer_nombre_afiliado(datos_del_nombre['texto_completo'])
                   
                   datos_del_nombre['cuil_afiliado'] = cuil_afiliado
                   datos_del_nombre['nombre_afiliado'] = nombre_afiliado
                   
               # Agregar file_id para posible uso posterior
               datos_del_nombre['file_id'] = file_id
               
               archivos_datos.append(datos_del_nombre)
               logger.info(f"  ✅ Datos extraídos de {nombre_original}")
           continue

       logger.info(f"\n==============================")
       logger.info(f"Procesando: {nombre_original}")

       request = service.files().get_media(fileId=file_id)
       fh = io.BytesIO()
       downloader = MediaIoBaseDownload(fh, request)
       done = False
       while not done:
           status, done = downloader.next_chunk()
       fh.seek(0)

       datos = None
       error = None
       
       if tipo == 'application/pdf':
           datos, error = extraer_desde_dos_paginas(fh, use_ai)
       elif tipo.startswith('image/'):
           datos, error = extraer_desde_imagen(fh.getvalue())
       else:
           resultados.append([nombre_original, '', '⚠️ Ignorado', 'Tipo no admitido'])
           continue

       if error:
           errores += 1
           resultados.append([nombre_original, '', '❌ ERROR', error])
           logger.info(f"  ❌ {error}")
           continue

       # Renombrar
       nuevo_nombre = datos['nombre_archivo']
       try:
           service.files().update(fileId=file_id, body={'name': nuevo_nombre}).execute()
           renombrados += 1
           resultados.append([nombre_original, nuevo_nombre, '✅ OK', ''])
           logger.info(f"  ✅ Renombrado a: {nuevo_nombre}")
           
           # Guardar datos para usar en generación de TXT
           datos['file_id'] = file_id
           datos['nombre_original'] = nombre_original
           archivos_datos.append(datos)
           
       except Exception as e:
           errores += 1
           resultados.append([nombre_original, '', '❌ ERROR', f'Error al renombrar: {e}'])
           logger.info(f"  ❌ Error al renombrar: {e}")

   # Generar log
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   log_filename = f'log_renombrado_{timestamp}.csv'
   with open(log_filename, 'w', newline='', encoding='utf-8') as f:
       writer = csv.writer(f)
       writer.writerow(['Nombre original', 'Nombre nuevo', 'Estado', 'Observación'])
       writer.writerows(resultados)

   # Subir log
   subir_log(service, log_filename, FOLDER_ID)

   logger.info(f"\n✅ Log guardado y subido como '{log_filename}'")
   logger.info(f"\n📄 Total de archivos revisados: {len(archivos)}")
   logger.info(f"✅ Renombrados correctamente: {renombrados}")
   logger.info(f"❌ Con errores: {errores}")
   
   return archivos_datos

# ========== FUNCIONES PARA GENERACIÓN DE TXT ==========
def obtener_excel(service):
   """Busca y descarga el archivo Excel desde Drive."""
   logger.info("Buscando archivo Excel en la carpeta...")
   
   # Buscar archivo con extensión .xlsx
   q = f"'{FOLDER_ID}' in parents and mimeType contains 'spreadsheet' and trashed = false"
   excel_files = service.files().list(
       q=q,
       fields="files(id, name)",
       pageSize=10
   ).execute().get('files', [])
   
   if not excel_files:
       logger.error("No se encontraron archivos Excel en la carpeta.")
       return None, None
   
   # Si hay varios Excel, preguntamos al usuario cuál usar
   if len(excel_files) > 1:
       logger.info(f"Se encontraron {len(excel_files)} archivos Excel:")
       for i, excel in enumerate(excel_files, 1):
           logger.info(f"{i}. {excel['name']}")
       
       seleccion = input("Ingrese el número del Excel a utilizar: ")
       try:
           idx = int(seleccion) - 1
           if idx < 0 or idx >= len(excel_files):
               raise ValueError("Índice fuera de rango")
           excel_file = excel_files[idx]
       except ValueError:
           logger.error("Selección inválida, utilizando el primer archivo.")
           excel_file = excel_files[0]
   else:
       excel_file = excel_files[0]
   
   logger.info(f"Descargando Excel: {excel_file['name']}")
   
   # Descargar el archivo Excel
   request = service.files().get_media(fileId=excel_file['id'])
   fh = io.BytesIO()
   downloader = MediaIoBaseDownload(fh, request)
   done = False
   while done is False:
       status, done = downloader.next_chunk()
       logger.info(f"Descarga {int(status.progress() * 100)}%")
   
   fh.seek(0)
   
   return fh, excel_file['name']

def leer_excel(excel_bytes):
   """Lee el archivo Excel y devuelve un DataFrame."""
   try:
       # Importar con parse_dates=False para evitar conversión automática de fechas
       df = pd.read_excel(excel_bytes, dtype=str, parse_dates=False)
       # Normalizar nombres de columnas
       df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
       
       # Verificar columnas requeridas
       columnas_requeridas = {"cuil", "codigo_certificado", "vencimiento_certificado", "provincia"}
       columnas_faltantes = columnas_requeridas - set(df.columns)
       
       if columnas_faltantes:
           logger.error(f"Faltan columnas en el Excel: {columnas_faltantes}")
           return None
       
       # Limpiar datos
       for col in ["cuil", "codigo_certificado", "provincia"]:
           df[col] = df[col].astype(str).str.strip()
           
       # Asegurar formato correcto de fecha de vencimiento
       if 'vencimiento_certificado' in df.columns:
           df['vencimiento_certificado'] = df['vencimiento_certificado'].apply(lambda x: formatear_fecha(x))
       
       # Convertir 'nombre_afiliado' si existe
       if 'nombre_afiliado' in df.columns:
           df.rename(columns={'nombre_afiliado': 'nombre'}, inplace=True)
       elif 'nombre' not in df.columns and 'nombre afiliado' in df.columns:
           df.rename(columns={'nombre afiliado': 'nombre'}, inplace=True)
       
       logger.info(f"Excel cargado con {len(df)} registros")
       return df
   
   except Exception as e:
       logger.error(f"Error al procesar el Excel: {e}")
       return None

def buscar_coincidencias_por_nombre(nombre_afiliado, df):
   """Busca coincidencias del nombre del afiliado en el Excel."""
   if not nombre_afiliado or 'nombre' not in df.columns:
       return None
   
   # Normalizar nombre (eliminar tildes, convertir a minúsculas)
   import unicodedata
   def normalizar(texto):
       if not isinstance(texto, str):
           return ""
       # Normalizar Unicode
       texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
       # Convertir a minúsculas
       return texto.lower()
   
   nombre_norm = normalizar(nombre_afiliado)
   
   # Crear columna normalizada para comparación
   df['nombre_norm'] = df['nombre'].apply(normalizar)
   
   # Buscar coincidencias exactas
   coincidencias = df[df['nombre_norm'] == nombre_norm]
   if not coincidencias.empty:
       return coincidencias.iloc[0]['cuil']
   
   # Si no hay coincidencias exactas, buscar coincidencias parciales
   # Dividir el nombre en palabras
   palabras = nombre_norm.split()
   if len(palabras) >= 2:
       # Buscar por apellido y primera inicial
       apellido = palabras[-1]  # Asumimos que el último es el apellido
       inicial = palabras[0][0] if palabras[0] else ""
       
       coincidencias = df[df['nombre_norm'].str.contains(f"{inicial}.*{apellido}", regex=True)]
       if not coincidencias.empty:
           return coincidencias.iloc[0]['cuil']
   
   return None

def generar_txt(service, archivos_datos, rnos="000000", use_ai=True):
   """
   Genera el archivo TXT basado en los archivos renombrados y el Excel.
   
   Args:
       service: Servicio de Google Drive autenticado
       archivos_datos: Lista de diccionarios con información de los PDFs procesados
       rnos: RNOS de la obra social (6 dígitos)
       use_ai: Indica si se debe usar IA para mejorar la extracción
   """
   logger.info("\n========== GENERANDO ARCHIVO TXT ==========")
   
   # Validar RNOS
   if not rnos.isdigit():
       rnos = ''.join(c for c in rnos if c.isdigit())
   rnos = rnos.zfill(6)[:6]
   logger.info(f"RNOS para generación de TXT: {rnos}")
   
   # 1. Obtener Excel
   excel_bytes, excel_name = obtener_excel(service)
   if not excel_bytes:
       logger.error("No se pudo obtener el archivo Excel.")
       return
       
   # 2. Leer Excel
   df = leer_excel(excel_bytes)
   if df is None:
       return
   
   # 3. Preparar datos del Excel para coincidencia
   
   # Asegurar que existe columna 'nombre' normalizada
   if 'nombre' not in df.columns and 'nombre_afiliado' in df.columns:
       df.rename(columns={'nombre_afiliado': 'nombre'}, inplace=True)
   elif 'nombre' not in df.columns and 'nombre afiliado' in df.columns:
       df.rename(columns={'nombre afiliado': 'nombre'}, inplace=True)
   
   # Normalizar nombres (quitar tildes, convertir a minúsculas)
   import unicodedata
   def normalizar(texto):
       if not isinstance(texto, str):
           return ""
       # Normalizar Unicode y convertir a minúsculas
       texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII').lower().strip()
       return texto
   
   # Crear columnas normalizadas
   if 'nombre' in df.columns:
       df['nombre_norm'] = df['nombre'].apply(normalizar)
   df['cuil_norm'] = df['cuil'].apply(lambda x: str(x).strip())
   
   # Crear diccionarios para búsqueda rápida
   nombres_a_cuil = {}
   palabras_a_cuil = {}  # Diccionario separado para palabras clave
   
   if 'nombre' in df.columns:
       for idx, fila in df.iterrows():
           nombre_norm = fila['nombre_norm']
           cuil = fila['cuil_norm']
           
           if nombre_norm:
               nombres_a_cuil[nombre_norm] = cuil  # Nombre completo -> CUIL
               # También añadir palabras individuales del nombre como claves
               palabras = nombre_norm.split()
               for palabra in palabras:
                   if len(palabra) > 3:  # Solo palabras con más de 3 caracteres
                       if palabra not in palabras_a_cuil:
                           palabras_a_cuil[palabra] = []
                       palabras_a_cuil[palabra].append(cuil)
   
   # 4. Extraer datos adicionales para TXT
   logger.info("Extrayendo datos adicionales para generación de TXT...")
   
   # Procesar facturas
   pdf_info = []  # Lista para guardar la info de todos los PDFs
   
   for i, datos in enumerate(archivos_datos, 1):
       logger.info(f"[{i}/{len(archivos_datos)}] Procesando {datos.get('nombre_archivo', 'archivo desconocido')}...")
       
       # Extraer datos adicionales
       info_completa = extraer_datos_adicionales(datos, use_ai)
       
       if not info_completa:
           logger.warning(f"  ⚠️ No se pudieron extraer datos adicionales")
           continue
       
       # Guardar información para coincidir después
       pdf_info.append(info_completa)
   
   # 5. Buscar coincidencias entre facturas y afiliados del Excel
   pdf_info_por_cuil = {}  # Diccionario final: CUIL del afiliado -> info factura
   
   # Primero, intentar coincidir por nombre de afiliado explícito
   for info in pdf_info:
       if 'nombre_afiliado' in info and info['nombre_afiliado']:
           nombre_norm = normalizar(info['nombre_afiliado'])
           
           # Búsqueda exacta
           if nombre_norm in nombres_a_cuil:
               cuil = nombres_a_cuil[nombre_norm]
               pdf_info_por_cuil[cuil] = info
               logger.info(f"  ✅ Coincidencia exacta por nombre: {nombre_norm} -> CUIL: {cuil}")
               continue
   
   # Segundo, buscar por palabras clave (nombres o apellidos)
   for info in pdf_info:
       # Omitir si ya se encontró coincidencia
       if any(info is v for v in pdf_info_por_cuil.values()):
           continue
           
       if 'nombre_afiliado' in info and info['nombre_afiliado']:
           nombre_norm = normalizar(info['nombre_afiliado'])
           palabras = nombre_norm.split()
           
           for palabra in palabras:
               if len(palabra) > 3 and palabra in palabras_a_cuil:
                   cuils = palabras_a_cuil[palabra]
                   if len(cuils) == 1:
                       cuil = cuils[0]  # Solo si hay una única coincidencia
                       pdf_info_por_cuil[cuil] = info
                       logger.info(f"  ✅ Coincidencia por palabra clave: {palabra} -> CUIL: {cuil}")
                       break
   
   # Tercero, buscar en todo el texto de la factura
   for info in pdf_info:
       # Omitir si ya se encontró coincidencia
       if any(info is v for v in pdf_info_por_cuil.values()):
           continue
           
       texto_factura = info.get('texto_factura', '')
       if not texto_factura:
           continue
       
       # Buscar nombres de afiliados en el texto
       for idx, fila in df.iterrows():
           if 'nombre' in fila and isinstance(fila['nombre'], str):
               nombre = fila['nombre'].lower()
               nombre_norm = fila['nombre_norm']
               cuil = fila['cuil_norm']
               
               # Si el nombre completo aparece en el texto de la factura
               if nombre in texto_factura or nombre_norm in texto_factura:
                   pdf_info_por_cuil[cuil] = info
                   logger.info(f"  ✅ Nombre encontrado en texto: {nombre} -> CUIL: {cuil}")
                   break
               
               # Si no coincide el nombre completo, probar con apellido
               if ' ' in nombre:
                   partes = nombre.split()
                   # Probar con el apellido (suponiendo que es la última parte)
                   apellido = partes[-1]
                   if len(apellido) > 3 and apellido in texto_factura:
                       # Verificar que no hay muchos otros afiliados con el mismo apellido
                       apellidos_coinciden = sum(1 for i, f in df.iterrows() 
                                             if 'nombre' in f and 
                                             isinstance(f['nombre'], str) and 
                                             f['nombre'].lower().split()[-1] == apellido)
                       
                       if apellidos_coinciden == 1:
                           pdf_info_por_cuil[cuil] = info
                           logger.info(f"  ✅ Apellido encontrado en texto: {apellido} -> CUIL: {cuil}")
                           break
   
   # 6. Mostrar estadísticas y ofrecer asignación manual
   coincidencias = len(pdf_info_por_cuil)
   total_afiliados = len(df)
   
   logger.info(f"\nCoincidencias encontradas: {coincidencias} de {total_afiliados} afiliados ({coincidencias*100/total_afiliados:.1f}%)")
   
   # Ofrecer asignación manual si no se encontraron suficientes coincidencias
   if coincidencias < min(20, total_afiliados * 0.15):  # Menos del 15% o menos de 20
       opcion = input("¿Desea establecer las correspondencias manualmente? (s/n): ")
       if opcion.lower() == 's':
           # Preparar listas ordenadas para selección
           pdfs_no_asignados = [info for info in pdf_info 
                               if not any(info is v for v in pdf_info_por_cuil.values())]
           
           afiliados_sin_factura = [fila for idx, fila in df.iterrows() 
                                   if fila['cuil_norm'] not in pdf_info_por_cuil]
           
           # Mostrar afiliados sin factura
           print("\nAfiliados sin factura asignada:")
           for i, fila in enumerate(afiliados_sin_factura, 1):
               nombre = fila.get('nombre', 'Sin nombre')
               cuil = fila['cuil_norm']
               print(f"{i}. {nombre} - CUIL: {cuil}")
           
           # Asignación manual optimizada
           seleccion_afiliado = int(input("\nSeleccione el número del afiliado: ")) - 1
           if 0 <= seleccion_afiliado < len(afiliados_sin_factura):
               afiliado = afiliados_sin_factura[seleccion_afiliado]
               
               # Mostrar facturas no asignadas
               print("\nFacturas disponibles:")
               for i, info in enumerate(pdfs_no_asignados, 1):
                   nombre_archivo = info.get('nombre_archivo', 'Desconocido')
                   cuit_prestador = info.get('cuit_pre', 'Sin CUIT')
                   nombre_afiliado = info.get('nombre_afiliado', '')
                   importe = info.get('importe', '0')
                   fecha = info.get('fecha_cbte', '')
                   print(f"{i}. {nombre_archivo} - Prestador: {cuit_prestador} - Importe: {importe} - Fecha: {fecha}")
                   if nombre_afiliado:
                       print(f"   Afiliado mencionado: {nombre_afiliado}")
               
               # Permitir múltiples asignaciones
               while True:
                   try:
                       seleccion_factura = input("\nSeleccione el número de la factura (0 para terminar): ")
                       if seleccion_factura == '0':
                           break
                           
                       seleccion_factura = int(seleccion_factura) - 1
                       if 0 <= seleccion_factura < len(pdfs_no_asignados):
                           info = pdfs_no_asignados[seleccion_factura]
                           cuil = afiliado['cuil_norm']
                           pdf_info_por_cuil[cuil] = info
                           print(f"Asignado: {info.get('nombre_archivo')} -> {afiliado.get('nombre', 'Afiliado')} ({cuil})")
                           
                           # Quitar la factura asignada de la lista
                           pdfs_no_asignados.pop(seleccion_factura)
                           
                           if not pdfs_no_asignados:
                               print("No quedan más facturas sin asignar.")
                               break
                       else:
                           print("Número de factura inválido.")
                   except ValueError:
                       print("Entrada inválida, debe ingresar un número.")
                   except Exception as e:
                       print(f"Error: {e}")
   
   # 7. Generar TXT
   logger.info("\nGenerando líneas para archivo TXT...")
   lineas_txt = []
   registros_procesados = 0
   registros_con_error = 0
   
   for idx, fila in df.iterrows():
       cuil = fila["cuil"].strip()
       
       # Buscar PDF correspondiente por CUIL del afiliado
       if cuil in pdf_info_por_cuil:
           info_pdf = pdf_info_por_cuil[cuil]
           linea = construir_linea(rnos, fila, info_pdf)
           
           if linea:
               lineas_txt.append(linea)
               registros_procesados += 1
               logger.info(f"Registro procesado: CUIL {cuil}")
           else:
               registros_con_error += 1
               logger.warning(f"No se pudo construir línea para CUIL {cuil}")
       else:
           registros_con_error += 1
           logger.warning(f"No se encontró PDF para CUIL {cuil}")
   
   # Verificar si se generaron líneas
   if not lineas_txt:
       logger.error("No se generó ninguna línea para el archivo TXT.")
       return
   
   # 8. Escribir TXT
   nombre_txt = f"{rnos}_ds.txt"
   with open(nombre_txt, "w", encoding="utf-8", newline="\r\n") as f:
       f.write("\n".join(lineas_txt))
   
   logger.info(f"Archivo TXT generado: {nombre_txt} ({len(lineas_txt)} líneas)")
   
   # Verificar formato antes de subir
   formato_correcto = verificar_formato_txt(nombre_txt)
   if not formato_correcto:
       logger.warning("El archivo TXT tiene errores de formato. Se recomienda revisar manualmente.")
   
   # 9. Subir TXT a Drive
   logger.info("Subiendo archivo TXT a Google Drive...")
   file_metadata = {'name': nombre_txt, 'parents': [FOLDER_ID], 'mimeType': 'text/plain'}
   media = MediaFileUpload(nombre_txt, mimetype='text/plain')
   subida = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
   logger.info(f"Archivo {nombre_txt} subido a Drive correctamente (ID: {subida.get('id')})")
   
   # 10. Resumen
   logger.info("\n" + "="*50)
   logger.info(f"RESUMEN DEL PROCESO:")
   logger.info(f"- Total de registros en Excel: {len(df)}")
   logger.info(f"- PDFs procesados correctamente: {len(archivos_datos)}")
   logger.info(f"- Registros procesados en TXT: {registros_procesados}")
   logger.info(f"- Registros con error: {registros_con_error}")
   logger.info("="*50)

# ========== EJECUCIÓN PRINCIPAL ==========
def main():
    """Función principal que ejecuta todo el proceso."""
    global DEVICE, MODEL_PATH # Asegúrate de que estas globales se usen o definan correctamente si son necesarias
    try:
        # Mostrar mensaje de bienvenida
        print("\n" + "="*50)
        print("   SISTEMA UNIFICADO DE FACTURAS - OBRA SOCIAL")
        print("="*50)
        print("Este programa realiza dos tareas:")
        print("1. Renombrar archivos PDF según patrón CUIT_CODIGO_PV_NRO")
        print("2. Generar archivo TXT para presentación en obra social")
        print("-"*50)
        
        # Preguntar si desea usar IA y forzar CPU si es necesario
        usar_ia_input = input("¿Desea utilizar IA para mejorar la extracción de datos? (s/n): ").lower()
        usar_ia = usar_ia_input == 's' # Convertir a booleano
        
        if usar_ia:
            forzar_cpu_input = input("¿Desea forzar el uso de CPU (útil si hay problemas con GPU)? (s/n): ").lower()
            if forzar_cpu_input == 's':
                DEVICE = torch.device("cpu") 
                logger.info("Forzando uso de CPU para el modelo de IA")
            
            # --- ESTA ES LA LÍNEA CLAVE MODIFICADA ---
            # Ruta correcta al modelo fine-tuneado, relativa a donde está prueba_bot_final.py
            path_modelo_fine_tuneado = os.path.join("fine_tuning_facturas", "layoutlmv3-finetuned-facturas_final")
            # -----------------------------------------

            usar_fine_tuned_input = input(f"¿Desea usar el modelo fine-tuneado desde '{path_modelo_fine_tuneado}'? (s/n): ").lower()
            if usar_fine_tuned_input == 's':
                if os.path.exists(path_modelo_fine_tuneado):
                    MODEL_PATH = path_modelo_fine_tuneado # Actualiza la variable global
                    logger.info(f"Usando modelo fine-tuneado desde: {MODEL_PATH}")
                else:
                    logger.warning(f"¡No se encontró el modelo fine-tuneado en {path_modelo_fine_tuneado}!")
                    continuar_base = input("El modelo fine-tuneado no está disponible. ¿Desea continuar con el modelo base 'microsoft/layoutlmv3-base'? (s/n): ").lower()
                    if continuar_base == 's':
                        MODEL_PATH = "microsoft/layoutlmv3-base" # Actualiza la variable global
                        logger.info(f"Usando modelo base por defecto: {MODEL_PATH}")
                    else:
                        logger.info("Proceso cancelado por el usuario al no encontrar modelo fine-tuneado.")
                        return 
            else:
                MODEL_PATH = "microsoft/layoutlmv3-base" # Actualiza la variable global
                logger.info(f"Usando modelo base por defecto: {MODEL_PATH}")
            
            inicializar_modelo() 
            logger.info("Sistema de IA inicializado correctamente")
        else:
            logger.info("Usando solo métodos tradicionales de extracción")
            factura_processor = None 
        
        logger.info("Autenticando con Google Drive...")
        service = autenticar()
        logger.info("Autenticación exitosa")
        
        logger.info("\n=== INICIANDO ETAPA 1: RENOMBRAR FACTURAS ===\n")
        archivos_datos = descargar_y_renombrar(service, usar_ia)
        
        if not archivos_datos:
            logger.warning("No se encontraron archivos PDF/Imagen para procesar o todos ya estaban renombrados inicialmente.")
            logger.info("Intentando buscar archivos ya renombrados para la generación de TXT...")
            
            archivos_drive = service.files().list(
                q=f"'{FOLDER_ID}' in parents and mimeType='application/pdf' and trashed = false",
                fields="files(id, name)",
                pageSize=1000 
            ).execute().get('files', [])
            
            archivos_ya_renombrados_encontrados = []
            if archivos_drive:
                for archivo_drive_info in archivos_drive:
                    nombre_archivo_drive = archivo_drive_info['name']
                    if re.match(PATTERN_RENAMED, nombre_archivo_drive):
                        datos_extraidos_nombre = extraer_datos_desde_nombre(nombre_archivo_drive)
                        if datos_extraidos_nombre:
                            datos_extraidos_nombre['file_id'] = archivo_drive_info['id']
                            archivos_ya_renombrados_encontrados.append(datos_extraidos_nombre)
            
            if archivos_ya_renombrados_encontrados:
                logger.info(f"Se encontraron {len(archivos_ya_renombrados_encontrados)} archivos ya renombrados que se usarán para el TXT.")
                archivos_datos = archivos_ya_renombrados_encontrados
            else:
                logger.warning("No hay archivos válidos para procesar para la generación de TXT.")
                input("\nPresione Enter para salir...")
                return 
        
        logger.info("\n=== INICIANDO ETAPA 2: GENERAR TXT ===\n")
        
        rnos_input = input("\nIngrese el RNOS de la obra social (6 dígitos): ").strip()
        generar_txt(service, archivos_datos, rnos_input, usar_ia) 
        
        logger.info("\n¡Proceso completado exitosamente!")
    
    except Exception as e: 
        logger.error(f"Error general en la ejecución de main: {e}", exc_info=True)
       
# ========== EJECUCIÓN DEL SCRIPT ==========
if __name__ == '__main__':
   try:
      main()
   except KeyboardInterrupt:
       logger.info("\nProceso interrumpido por el usuario.")
   except Exception as e:
       # Este except es para errores que podrían ocurrir fuera de main(),
       # aunque es menos probable si casi todo está dentro de main().
       logger.error(f"Error inesperado fuera de la función main: {e}", exc_info=True)
   finally:
       input("\nPresione Enter para salir...")