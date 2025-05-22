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
# from transformers import AdamW # Comentado si no se usa directamente aquí
import numpy as np
# import cv2 # Comentado si no se usa directamente aquí, aunque el procesador de LayoutLMv3 podría necesitarlo internamente
from tqdm import tqdm # Comentado si no se usa directamente aquí

# ========== CONFIGURACIÓN ==========
# Ajusta estas rutas según tu configuración
FOLDER_ID = '1-pMVR5Nh4k_Jlenygaju0R1nH0YcGTFP'  # Carpeta con PDFs
SCOPES = ['https://www.googleapis.com/auth/drive']

# Configurar OCR
TESSERACT_CMD = 'tesseract' # Ajusta si es necesario, ej: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Configurar logging
logging.basicConfig(
    level=logging.DEBUG,  # CAMBIADO a DEBUG para ver los nuevos logs
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', # Formato más detallado
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("facturas_bot.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configuraciones específicas para generación de TXT
PATTERN_RENAMED = r'^([0-9]{11})_([1-9]\d*)_([1-9]\d*)_([1-9]\d*)\.pdf$'
CODIGO_CBTE = {
    ("FACTURA", "B"): "03", # Debe ser ("B", "FACTURA") o normalizar el orden al buscar
    ("RECIBO", "B"): "04",  # Debe ser ("B", "RECIBO")
    ("FACTURA", "C"): "05", # Debe ser ("C", "FACTURA")
    ("RECIBO", "C"): "06",  # Debe ser ("C", "RECIBO")
}
# CORRECCIÓN EN CODIGO_CBTE para que coincida con la lógica de tipo_y_letra_a_codigo
CODIGO_CBTE_CORREGIDO = { 
    ("B", "FACTURA"): "03",
    ("B", "RECIBO"): "04",
    ("C", "FACTURA"): "05",
    ("C", "RECIBO"): "06",
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
# Ruta al modelo fine-tuneado
MODEL_PATH = "./layoutlmv3-finetuned-facturas_final"  # Ruta por defecto, se puede cambiar en main()
MAX_LENGTH = 512  # Máxima longitud de tokens para procesar
BATCH_SIZE = 1  # Ajustar según la memoria de GPU

# Verificar GPU
def check_gpu():
    if torch.cuda.is_available():
        logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Nota: memory_reserved da la memoria reservada por el caché de PyTorch, no necesariamente la "disponible" total
        # logger.info(f"Memoria GPU reservada por PyTorch: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        # logger.info(f"Memoria GPU asignada por PyTorch: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    else:
        logger.info("GPU no disponible, usando CPU")

# ========== CLASES Y FUNCIONES PARA IA ==========
class FacturaProcessor:
    """Clase para procesar facturas con LayoutLMv3"""
    
    def __init__(self, model_path=MODEL_PATH):
        """Inicializa el procesador y los modelos"""
        logger.info(f"Inicializando FacturaProcessor con model_path: {model_path}")
        self.model_path_internal = model_path # Guardar la ruta para usarla en load_entity_extractor

        try:
            # Intentar cargar el procesador entrenado
            self.processor = LayoutLMv3Processor.from_pretrained(self.model_path_internal)
            logger.info(f"Procesador LayoutLMv3 cargado exitosamente desde: {self.model_path_internal}")
        except Exception as e:
            logger.warning(f"No se pudo cargar el procesador entrenado desde {self.model_path_internal}: {e}")
            # Cargar procesador base como respaldo
            self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
            logger.info("Procesador LayoutLMv3 base cargado como RESPALDO.")
        
        # Cargar modelo para clasificación de tipo de documento
        self.doc_classifier = None  # Se cargará bajo demanda para ahorrar memoria
        
        # Cargar modelo para extraer entidades
        self.entity_extractor = None  # Se cargará bajo demanda para ahorrar memoria
        
        # Lista de entidades que queremos extraer
        self.entity_labels = [
            "CUIT_PRESTADOR", "CUIT_AFILIADO", "NOMBRE_AFILIADO", "NOMBRE_PRESTADOR",
            "TIPO_FACTURA", "LETRA_FACTURA", "PUNTO_VENTA", "NUMERO_FACTURA",
            "FECHA_EMISION", "CAE", "IMPORTE", "PERIODO", "ACTIVIDAD", "DNI_AFILIADO"
        ] #
        
        logger.info("FacturaProcessor inicializado.")
    
    def load_doc_classifier(self):
        """Carga el modelo de clasificación de documentos bajo demanda"""
        if self.doc_classifier is None:
            logger.info(f"Cargando modelo de clasificación de documentos desde {self.model_path_internal} (o base si falla)...")
            try:
                self.doc_classifier = LayoutLMv3ForSequenceClassification.from_pretrained(
                    self.model_path_internal, 
                    num_labels=2  # Factura válida o no válida
                )
                logger.info(f"Modelo de clasificación cargado exitosamente desde {self.model_path_internal}")
            except Exception as e:
                logger.warning(f"No se pudo cargar el modelo de clasificación desde {self.model_path_internal}: {e}")
                self.doc_classifier = LayoutLMv3ForSequenceClassification.from_pretrained(
                    "microsoft/layoutlmv3-base", 
                    num_labels=2
                )
                logger.info("Modelo de clasificación base 'microsoft/layoutlmv3-base' cargado como RESPALDO.")
            self.doc_classifier.to(DEVICE)
            # logger.info("Modelo de clasificación movido a DEVICE.") # Ya se loggea si es GPU o CPU
    
    def load_entity_extractor(self):
        """Carga el modelo de extracción de entidades bajo demanda"""
        if self.entity_extractor is None:
            logger.info(f"Cargando modelo de extracción de entidades desde {self.model_path_internal}...")
            
            try:
                # Intenta cargar el modelo fine-tuneado
                self.entity_extractor = LayoutLMv3ForTokenClassification.from_pretrained(
                    self.model_path_internal 
                )
                logger.info(f"Modelo de extracción de entidades fine-tuneado cargado exitosamente desde: {self.model_path_internal}")
            except Exception as e:
                logger.error(f"Error al cargar el modelo fine-tuneado desde {self.model_path_internal}: {e}")
                logger.info("Intentando cargar el modelo base 'microsoft/layoutlmv3-base' como respaldo...")
                # Carga modelo base como respaldo
                # Es importante que num_labels coincida con lo que el modelo base puede manejar o lo que definas aquí
                # Si el base no fue entrenado para tus etiquetas, esto será problemático
                # Para un modelo base genérico, num_labels se inferiría de su config si no se especifica, 
                # o se debe especificar si se quiere una cabeza nueva (pero aquí solo cargamos preentrenado)
                # Lo más seguro es que si el fine-tuneado falla, la extracción de tus entidades específicas no funcionará bien con el base.
                self.entity_extractor = LayoutLMv3ForTokenClassification.from_pretrained(
                    "microsoft/layoutlmv3-base"
                    # num_labels=len(self.entity_labels) * 2 + 1 # Esto sería para entrenar una nueva cabeza
                )
                logger.info("Modelo de extracción de entidades base 'microsoft/layoutlmv3-base' cargado como RESPALDO.")
            
            self.entity_extractor.to(DEVICE)
            # logger.info("Modelo de extracción de entidades movido a DEVICE.")
    
    def preprocess_document(self, image):
        """Preprocesa una imagen para el modelo LayoutLMv3"""
        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Extraer texto con pytesseract
        # Asegúrate que TESSERACT_CMD esté bien configurado globalmente o pásalo aquí
        ocr_result = pytesseract.image_to_data(image, lang='spa', output_type=pytesseract.Output.DICT) #
        
        # Extraer palabras y coordenadas
        words = []
        boxes = []
        for i in range(len(ocr_result["text"])):
            word = ocr_result["text"][i].strip()
            if word: # Solo si la palabra no está vacía
                words.append(word)
                x = ocr_result["left"][i]
                y = ocr_result["top"][i]
                w = ocr_result["width"][i]
                h = ocr_result["height"][i]
                # Las coordenadas que espera LayoutLMv3Processor son [x_min, y_min, x_max, y_max]
                # y deben estar normalizadas a 0-1000. Pytesseract da x,y,w,h en píxeles.
                # El procesador de Hugging Face se encarga de la normalización si le pasas los boxes en píxeles.
                boxes.append([x, y, x + w, y + h])
        
        if not words: # Si no se extrajeron palabras
            logger.warning("OCR no extrajo ninguna palabra de la imagen.")
            # Devolver un encoding vacío o con valores por defecto para evitar errores posteriores
            # Esto es importante para que el modelo no reciba tensores vacíos o malformados
            # Crear un encoding dummy con la estructura esperada pero datos que no causen error
            # Necesitaríamos saber la estructura exacta que espera el modelo.
            # Por simplicidad, si no hay palabras, devolvemos None y quien llame debe manejarlo.
            # O mejor, intentamos pasar la imagen sola. El procesador puede hacer OCR si se le dice.
            # El actual `LayoutLMv3Processor` usa `apply_ocr=False` si se carga el base, 
            # pero el fine-tuneado podría tenerlo diferente.
            # Si el procesador fine-tuneado fue guardado con `apply_ocr=True` por defecto,
            # entonces no necesitaría `words` y `boxes` explícitamente aquí.
            # Pero el código actual SÍ los pasa.
            logger.info("Intentando procesar la imagen sin palabras/cajas explícitas (depende de la config del procesador)")
            # encoding = self.processor(image, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            # Lo anterior es si el procesador hace OCR. Como le pasamos words y boxes, si están vacíos puede fallar.
            # Si words está vacío, el procesador puede dar error.
            # Es mejor asegurarse que words y boxes tengan al menos un elemento dummy si es necesario
            # o que el procesador esté configurado para hacer OCR si no se le dan.
            # Por ahora, si no hay palabras, la codificación fallará o dará resultados inesperados.
            # El código actual seguirá y es probable que encoding sea problemático.

        # Codificar para el modelo
        # El procesador de LayoutLMv3 espera que las coordenadas estén normalizadas a 0-1000.
        # Sin embargo, si le pasas las coordenadas en píxeles, el procesador *debería* normalizarlas
        # si la imagen también se proporciona.
        encoding = self.processor(
            image, 
            text=words, # El argumento es 'text' o 'tokenized_text' no 'words'
            boxes=boxes, 
            truncation=True, 
            padding="max_length", # Añadido padding para asegurar longitud uniforme
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
        if self.doc_classifier is None: # Si no se pudo cargar
            logger.warning("Clasificador de documentos no disponible. Asumiendo que NO es una factura válida.")
            return False, 0.0

        try:
            encoding, _, _ = self.preprocess_document(image)
            if not encoding['input_ids'].numel(): # Si el encoding está vacío
                 logger.warning("Encoding vacío en is_valid_factura. No se puede clasificar.")
                 return False, 0.0
        except Exception as e:
            logger.error(f"Error en preprocess_document para is_valid_factura: {e}")
            return False, 0.0 # No se puede procesar, asumir no válido

        # Clasificar el documento
        with torch.no_grad():
            outputs = self.doc_classifier(**encoding)
            predictions_softmax = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction_idx = torch.argmax(predictions_softmax, dim=1).item()
            confidence = predictions_softmax[0][prediction_idx].item()
        
        # Asumimos que 1 es "factura válida" y 0 es "no válida"
        # Esto depende de cómo se entrenó el clasificador y el orden de las etiquetas.
        # Sería mejor tener las etiquetas del clasificador guardadas.
        return prediction_idx == 1, confidence
    
    def extract_entities(self, image):
        """Extrae entidades de una factura usando el modelo fine-tuneado"""
        self.load_entity_extractor()
        if self.entity_extractor is None: # Si no se pudo cargar
            logger.error("Extractor de entidades no disponible.")
            return {}

        try:
            encoding, words, boxes = self.preprocess_document(image)
            if not encoding['input_ids'].numel(): # Si el encoding está vacío
                 logger.warning("Encoding vacío en extract_entities. No se pueden extraer entidades.")
                 return {}
        except Exception as e:
            logger.error(f"Error en preprocess_document para extract_entities: {e}")
            return {} # No se puede procesar

        # === INICIO DE CÓDIGO DE DEPURACIÓN AÑADIDO ===
        logger.debug("--- Iniciando depuración de extract_entities ---")
        input_ids = encoding.get('input_ids') # Usar .get() para evitar KeyError si falta
        if input_ids is not None:
            logger.debug(f"Shape of input_ids: {input_ids.shape}")
            if input_ids.numel() > 0: # Solo calcular min/max si no está vacío
                logger.debug(f"Min input_id: {input_ids.min().item()}")
                logger.debug(f"Max input_id: {input_ids.max().item()}")
            else:
                logger.debug("input_ids está vacío.")
        else:
            logger.debug("input_ids no encontrado en encoding.")

        # Obtener el vocab_size del config del modelo cargado
        # Es importante que self.entity_extractor esté cargado aquí
        if self.entity_extractor and hasattr(self.entity_extractor, 'config'):
            vocab_size = self.entity_extractor.config.vocab_size
            logger.debug(f"Model vocab_size: {vocab_size}")

            if input_ids is not None and input_ids.numel() > 0 and input_ids.max().item() >= vocab_size:
                logger.error(f"¡ALERTA CRÍTICA! Max input_id ({input_ids.max().item()}) es >= que vocab_size ({vocab_size}). Esto causará un error de CUDA.")
            
            if input_ids is not None and input_ids.numel() > 0 and input_ids.min().item() < 0:
                # Es normal que haya IDs negativos si corresponden al padding_idx
                # Necesitamos el pad_token_id real del tokenizer
                pad_token_id_from_processor = self.processor.tokenizer.pad_token_id if self.processor and hasattr(self.processor, 'tokenizer') else None
                logger.debug(f"Min input_id es negativo: {input_ids.min().item()}. pad_token_id del procesador: {pad_token_id_from_processor}")
                # Aquí no hay una alerta de error inmediata porque el padding puede ser negativo si se configura así.
                # Lo importante es que no sea un ID negativo inesperado.
        else:
            logger.error("No se pudo acceder a self.entity_extractor.config.vocab_size para depuración.")
        logger.debug("--- Fin de depuración de extract_entities (antes de la llamada al modelo) ---")
        # === FIN DE CÓDIGO DE DEPURACIÓN AÑADIDO ===

        # Extraer entidades
        with torch.no_grad():
            outputs = self.entity_extractor(**encoding) # Esta es la línea donde probablemente ocurre el error de CUDA
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        # Procesar resultados
        results = {}
        current_entity = None
        current_text = []
        
        # Crear mapeo de IDs a etiquetas según la configuración del modelo
        # Es más robusto usar id2label del config del modelo si está disponible
        id2label_from_model_config = self.entity_extractor.config.id2label if hasattr(self.entity_extractor, 'config') and hasattr(self.entity_extractor.config, 'id2label') else None

        if id2label_from_model_config:
            id2label = {int(k): v for k,v in id2label_from_model_config.items()} # Asegurar que las claves sean enteros
            logger.debug(f"Usando id2label desde config del modelo: {id2label}")
        else:
            # Fallback a la construcción manual si no está en config (esto es menos ideal)
            logger.warning("id2label no encontrado en config del modelo. Construyendo manualmente (esto podría ser propenso a errores).")
            label_list = ["O"] 
            for label_name in self.entity_labels:
                label_list.append(f"B-{label_name}")
                label_list.append(f"I-{label_name}")
            id2label = {i: label for i, label in enumerate(label_list)}
            logger.debug(f"id2label construido manualmente: {id2label}")

        max_valid_prediction_id = len(id2label) - 1

        for idx, prediction_id in enumerate(predictions): # predictions ya es [0] así que es 1D
            if idx >= len(words): # Evitar IndexError si hay más predicciones que palabras (raro pero posible con padding/trunc)
                logger.warning(f"Índice de predicción {idx} fuera de rango para la cantidad de palabras ({len(words)}). Deteniendo procesamiento de etiquetas para esta instancia.")
                break
            word = words[idx] # Palabra correspondiente a esta predicción de token

            if prediction_id > max_valid_prediction_id or prediction_id < 0:
                logger.warning(f"Predicción con ID {prediction_id} fuera de rango para id2label (max ID: {max_valid_prediction_id}). Palabra: '{word}'. Se tratará como 'O'.")
                label = "O" # Tratar como "O" para evitar error
            else:
                label = id2label.get(prediction_id, "O") # Usar .get con default "O" por si acaso
            
            if label == "O":
                if current_entity and current_text:
                    results[current_entity] = " ".join(current_text)
                current_entity = None
                current_text = []
            elif label.startswith("B-"):
                if current_entity and current_text: # Guardar entidad anterior
                    results[current_entity] = " ".join(current_text)
                current_entity = label[2:]
                current_text = [word]
            elif label.startswith("I-"):
                extracted_entity_type = label[2:]
                if current_entity == extracted_entity_type: # Continuar con la entidad actual
                    current_text.append(word)
                else: # Etiqueta I- pero no coincide con la B- anterior, tratar como nueva B- o como O
                    if current_entity and current_text: # Guardar entidad anterior
                         results[current_entity] = " ".join(current_text)
                    # Podríamos iniciar una nueva entidad aquí si quisiéramos ser más permisivos
                    # current_entity = extracted_entity_type 
                    # current_text = [word]
                    # O, más conservador, resetear:
                    current_entity = None 
                    current_text = []
                    logger.warning(f"Etiqueta 'I-{extracted_entity_type}' encontrada sin 'B-{extracted_entity_type}' previa o para una entidad diferente ('{current_entity}'). Palabra: '{word}'. Tratando como O.")


        # No olvidar la última entidad
        if current_entity and current_text:
            results[current_entity] = " ".join(current_text)
        
        return results
    
    def verificar_predicciones(self, entities):
        """Verifica la validez de las predicciones del modelo"""
        confianza = 0
        # Verificar entidades críticas
        entidades_criticas = ["CUIT_PRESTADOR", "PUNTO_VENTA", "NUMERO_FACTURA"] #
        for entidad in entidades_criticas:
            if entidad in entities and entities[entidad]: # Asegurar que la entidad existe y no está vacía
                valor_entidad = str(entities[entidad]) # Convertir a string por si acaso
                # Verificaciones específicas por tipo de entidad
                if entidad == "CUIT_PRESTADOR" and len(valor_entidad.replace("-", "").replace(" ", "")) == 11:
                    confianza += 0.33
                elif entidad == "PUNTO_VENTA" and any(c.isdigit() for c in valor_entidad):
                    confianza += 0.33
                elif entidad == "NUMERO_FACTURA" and any(c.isdigit() for c in valor_entidad):
                    confianza += 0.33
        
        # Devolver nivel de confianza y mensaje
        if confianza >= 0.9: # Ajustado el umbral, 0.33*3 = 0.99
            return True, f"Predicciones de alta confianza ({confianza:.2f})"
        elif confianza >= 0.66:
            return True, f"Predicciones aceptables ({confianza:.2f})"
        else:
            return False, f"Predicciones de baja confianza ({confianza:.2f})"

# Crear una instancia global del procesador
factura_processor = None

def inicializar_modelo():
    """Inicializa el modelo de IA"""
    global factura_processor, MODEL_PATH # Asegurarse que MODEL_PATH global se usa aquí
    check_gpu()
    logger.info(f"En inicializar_modelo, usando MODEL_PATH: {MODEL_PATH}")
    factura_processor = FacturaProcessor(model_path=MODEL_PATH) # Pasar la ruta del modelo

def registrar_resultados_modelo(entities, metodo_final, texto_factura):
    """Registra los resultados del modelo para análisis posterior"""
    try:
        # Crear archivo si no existe
        if not os.path.exists('resultados_modelo.csv'):
            with open('resultados_modelo.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Método_Final', 'Entidades_Encontradas', 'Texto_Factura_Inicio'])
        
        # Registrar resultado
        with open('resultados_modelo.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                metodo_final,
                # Convertir dict_keys a lista y luego a string para mejor legibilidad
                str(list(entities.keys())) if isinstance(entities, dict) else str(entities), 
                texto_factura[:500] if texto_factura else ""  # Primeros 500 caracteres
            ])
    except Exception as e:
        logger.error(f"Error al registrar resultados del modelo: {e}")

# ========== FUNCIONES DE AUTENTICACIÓN ==========
def autenticar():
    """Autenticación con Google Drive API."""
    creds = None
    token_file = 'token.json'
    credentials_file = 'credentials.json'

    if not os.path.exists(credentials_file):
        logger.error(f"Archivo '{credentials_file}' no encontrado. Por favor, descárgalo desde Google Cloud Console y colócalo en el mismo directorio que este script.")
        raise FileNotFoundError(f"Se requiere '{credentials_file}' para la autenticación.")

    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        except Exception as e:
            logger.warning(f"Error al cargar '{token_file}': {e}. Se intentará re-autenticar.")
            creds = None # Forzar re-autenticación

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("Refrescando token de acceso...")
                creds.refresh(Request())
                logger.info("Token refrescado exitosamente.")
            except Exception as e:
                logger.warning(f"Error al refrescar el token: {e}. Se eliminará '{token_file}' y se intentará re-autenticar.")
                if os.path.exists(token_file):
                    os.remove(token_file)
                creds = None # Forzar re-autenticación completa
        else: # No hay token, o no es válido y no se puede refrescar
            logger.info(f"No hay credenciales válidas o '{token_file}' no existe/es inválido. Iniciando flujo de autenticación.")
            try:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
                logger.info("Autenticación completada.")
            except Exception as e:
                logger.error(f"Error durante el flujo de autenticación: {e}")
                raise Exception(f"No se pudo autenticar con Google Drive: {e}")

        # Guardar las credenciales para la próxima ejecución
        try:
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
            logger.info(f"Credenciales guardadas en '{token_file}'.")
        except Exception as e:
            logger.error(f"Error al guardar el token en '{token_file}': {e}")
            
    return build('drive', 'v3', credentials=creds)


# ========== FUNCIONES PARA RENOMBRAR FACTURAS ==========
def limpiar_numero(num_str):
    """Elimina ceros a la izquierda de un número."""
    try:
        num_str_cleaned = ''.join(c for c in str(num_str) if c.isdigit())
        return str(int(num_str_cleaned)) if num_str_cleaned else "0"
    except ValueError: # Si no se puede convertir a int (ej. cadena vacía después de limpiar)
        return str(num_str) if num_str else "0" # Devolver original o "0"

def extraer_cuits(texto):
    """Extrae todos los CUITs/CUILs presentes en el texto."""
    cuits = []
    patrones = [
        r"\bCUIT:?\s*(\d{2}-\d{8}-\d)\b",
        r"\bCUIT:?\s*(\d{11})\b",
        r"\bCUIL:?\s*(\d{2}-\d{8}-\d)\b",
        r"\bCUIL:?\s*(\d{11})\b",
        r"\b(\d{2}-\d{8}-\d)\b", 
        r"\b(\d{11})\b" 
    ] #
    
    for patron in patrones:
        matches = re.finditer(patron, texto, re.I)
        for match in matches:
            cuit_match = match.group(1).replace("-", "")
            if len(cuit_match) == 11 and cuit_match not in cuits: # Evitar duplicados
                cuits.append(cuit_match)
    
    return cuits

def extraer_cuit(texto): # CUIT del prestador
    """Extrae el CUIT del prestador del texto de la factura."""
    # Priorizar CUITs que estén claramente etiquetados o en posiciones típicas de prestador
    # Ejemplo: Buscar CUIT al principio, o asociado a "Razón Social", "Prestador", etc.
    # Esta función es un poco genérica y podría mejorarse con más contexto de la factura
    
    # Patrón más específico para CUIT, buscando 11 dígitos o formato con guiones
    match = re.search(r'\b(\d{2}-?\d{8}-?\d{1})\b', texto) 
    if match:
        return match.group(1).replace("-", "")
    
    # Si no, tomar el primer CUIT encontrado por la función más general
    # (asumiendo que el del prestador suele aparecer primero)
    todos_cuits = extraer_cuits(texto)
    if todos_cuits:
        return todos_cuits[0]
        
    return None

def extraer_cuil_afiliado(texto):
    """Extrae el CUIL del afiliado/beneficiario del texto de la factura."""
    patrones = [
        r"(?:Paciente|Beneficiario|Afiliado|Cliente)[\s:]+.*?(?:DNI|CUIL|N° Socio)[:\s]+.*?(\d{2}[-\s]?\d{8}[-\s]?\d{1}\b|\d{11}\b)",
        r"(?:Paciente|Beneficiario|Afiliado|Cliente)[:\s]+.*?(\d{2}[-\s]?\d{8}[-\s]?\d{1}\b|\d{11}\b)",
        r"Datos del (?:paciente|beneficiario|afiliado|cliente)[:\s]+.*?CUIL[:\s]+(\d{2}[-\s]?\d{8}[-\s]?\d{1}\b|\d{11}\b)",
        r"(?:CUIL|DNI) del (?:paciente|beneficiario|afiliado|cliente)[:\s]+(\d{2}[-\s]?\d{8}[-\s]?\d{1}\b|\d{11}\b)"
    ] #
    
    for patron in patrones:
        match = re.search(patron, texto, re.IGNORECASE | re.DOTALL)
        if match:
            cuil = match.group(1).replace("-", "").replace(" ", "")
            if len(cuil) == 11:
                return cuil
    
    # Si no encuentra con los patrones específicos, buscar todos los CUIT/CUIL
    # y si hay más de uno, intentar devolver el segundo (asumiendo que el primero es el prestador)
    # Esto es riesgoso si no hay garantía sobre el orden o cantidad de CUITs
    # Aquí sería mejor si la IA puede distinguir CUIT_PRESTADOR de CUIT_AFILIADO
    cuits_encontrados = extraer_cuits(texto)
    cuit_prestador = extraer_cuit(texto) # Obtener el CUIT del prestador para excluirlo

    for cuit_candidato in cuits_encontrados:
        if cuit_candidato != cuit_prestador:
            return cuit_candidato # Devolver el primer CUIT que no sea el del prestador
            
    return None

def extraer_nombre_afiliado(texto):
    """Extrae el nombre del afiliado del texto de la factura."""
    patrones = [
        r"(?:Paciente|Beneficiario|Afiliado|Cliente)[:\s]+([A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s,]+)(?:DNI|CUIL|N° Socio)",
        r"Datos del (?:paciente|beneficiario|afiliado|cliente)[:\s]+Nombre[:\s]+([A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s,]+)",
        r"Nombre del (?:paciente|beneficiario|afiliado|cliente)[:\s]+([A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s,]+)"
    ] #
    
    for patron in patrones:
        match = re.search(patron, texto, re.IGNORECASE | re.DOTALL)
        if match:
            nombre = match.group(1).strip()
            # Limpiar posible "Apellido, Nombre" a "Nombre Apellido"
            if "," in nombre:
                partes = [p.strip() for p in nombre.split(",")]
                if len(partes) == 2:
                    nombre = f"{partes[1]} {partes[0]}"
            if len(nombre) > 3 and any(c.isalpha() for c in nombre): # Verificar que sea un nombre válido
                return nombre.title() # Capitalizar
    
    return None

def extraer_tipo_y_letra(texto):
    """Extrae el tipo de comprobante (FACTURA/RECIBO) y letra (B/C)."""
    tipo_match = re.search(r'(FACTURA|RECIBO|NOTA DE CREDITO|NOTA DE DEBITO)', texto, re.IGNORECASE) # Ampliado
    tipo = tipo_match.group(1).upper() if tipo_match else None

    if not tipo: return None, None

    # Convertir "NOTA DE CREDITO" a "FACTURA" o "RECIBO" si es necesario para CODIGO_CBTE
    # O manejar códigos específicos para notas de crédito/débito si se añaden a CODIGO_CBTE_CORREGIDO
    # Por ahora, simplificamos: si es NOTA, intentar con letra.
    # El código AFIP es diferente para notas de crédito/débito.
    # Ejemplo: 08 para Nota de Crédito C, 07 para Nota de Débito C
    # Si se quiere manejar esto, CODIGO_CBTE_CORREGIDO debe expandirse.

    # Intentar extraer la letra después del tipo de comprobante
    if tipo_match:
        texto_despues_tipo = texto[tipo_match.end():]
        letra_contexto_match = re.search(r"^\s*([BC])\b", texto_despues_tipo, re.IGNORECASE) # Letra justo después
        if letra_contexto_match:
            return tipo, letra_contexto_match.group(1).upper()

    letra_cod = re.search(r'\b([BC])\s*COD\.?\s*\d*', texto, re.IGNORECASE)
    if letra_cod:
        return tipo, letra_cod.group(1).upper()

    letra_suelta_aislada = re.search(r'(?:^|\s)\s*([BC])\s*(?:$|\s)', texto, re.MULTILINE | re.IGNORECASE)
    if letra_suelta_aislada:
         # Verificar que no sea parte de una palabra más larga, ej. "Calle B..."
        # Esta regex es un poco riesgosa.
        # Podríamos verificar el contexto alrededor.
        # Por ahora, la tomamos si aparece.
        return tipo, letra_suelta_aislada.group(1).upper()


    letra_numerica = re.search(r'\b([BC])\d{2,4}\b', texto, re.IGNORECASE)
    if letra_numerica:
        return tipo, letra_numerica.group(1).upper()
    
    # Fallback para tipo "FACTURA" si no se encontró letra explícita
    # Muchas facturas C no dicen la letra C prominentemente.
    if tipo == "FACTURA": # Podríamos asumir C si no se encuentra B
        if not re.search(r"\bB\b", texto, re.IGNORECASE): # Si no hay una B clara
            return tipo, "C" # Asumir C por defecto para FACTURA si no hay otra indicación

    return tipo, None # Devolver tipo aunque no se encuentre letra

def tipo_y_letra_a_codigo(tipo, letra):
    """Convierte tipo y letra a código de comprobante AFIP."""
    if not tipo or not letra:
        return None
    
    tipo_upper = tipo.upper()
    letra_upper = letra.upper()

    # Manejar casos donde tipo puede ser más específico que solo "FACTURA" o "RECIBO"
    # Para este diccionario, necesitamos "FACTURA" o "RECIBO"
    if "FACTURA" in tipo_upper:
        tipo_base = "FACTURA"
    elif "RECIBO" in tipo_upper:
        tipo_base = "RECIBO"
    # Aquí se podrían añadir más mapeos si el OCR extrae "NOTA DE CREDITO", etc.
    # y si CODIGO_CBTE_CORREGIDO se expande para esos tipos.
    else:
        tipo_base = tipo_upper # Usar tal cual si no es factura ni recibo (puede fallar la búsqueda)

    key = (letra_upper, tipo_base)
    return int(CODIGO_CBTE_CORREGIDO.get(key, 0)) # Usar el corregido y default a 0 o None

def extraer_pv_nro(texto):
    """Extrae punto de venta y número de comprobante."""
    patrones = [
        r'Comp\.?\s*Nro\.?:\s*0*(\d+)\s+0*(\d+)', # Comp. Nro: 00000 00000000
        r'Nro\.?\s*Comprobante:\s*0*(\d+)\s*-\s*0*(\d+)', # Nro Comprobante: 00000-00000000
        r'Nro\.?\s*0*(\d+)\s*-\s*0*(\d+)',           # Nro 00004-00003575
        r'\b0*(\d{1,5})\s*-\s*0*(\d{1,8})\b',        # 0002-00027842 (más genérico)
        r'(?:FAC-|[BC])\s*-\s*0*(\d+)\s*-\s*0*(\d+)', # (FAC-)?B - 0003 - 00002475
        r'(?:FAC-|[BC])\-0*(\d+)\-0*(\d+)'        # (FAC-)?B-0003-00002475
    ] #
    
    for patron in patrones:
        match = re.search(patron, texto, re.IGNORECASE | re.DOTALL)
        if match:
            # Algunos patrones capturan 2 grupos, otros pueden capturar más si tienen prefijos
            # Nos interesan los dos últimos grupos que contienen números
            grupos_numericos = [g for g in match.groups() if g and g.isdigit()]
            if len(grupos_numericos) >= 2:
                 # Tomar los dos últimos si hay más de dos (por ejemplo, si la letra fue capturada)
                pv = limpiar_numero(grupos_numericos[-2])
                nro = limpiar_numero(grupos_numericos[-1])
                # Validación simple de longitud (PV hasta 5, NRO hasta 8)
                if len(pv) <= 5 and len(nro) <= 8:
                    return pv, nro
            elif len(match.groups()) == 2: # Caso estándar de 2 grupos
                pv = limpiar_numero(match.group(1))
                nro = limpiar_numero(match.group(2))
                if len(pv) <= 5 and len(nro) <= 8:
                    return pv, nro


    return None, None


def extraer_datos_flexible(texto):
    """Extrae todos los datos necesarios para renombrar el archivo."""
    cuit = extraer_cuit(texto)
    # if not cuit: return None # Comentado para intentar extraer otros datos aunque falte CUIT inicialmente

    cuil_afiliado = extraer_cuil_afiliado(texto)
    nombre_afiliado = extraer_nombre_afiliado(texto)
    
    tipo, letra = extraer_tipo_y_letra(texto)
    codigo = tipo_y_letra_a_codigo(tipo, letra) if tipo and letra else None

    pv, nro = extraer_pv_nro(texto)

    if not (cuit and codigo and pv and nro):
        # Si falta alguno de los datos críticos para el nombre, no podemos renombrar
        # Podríamos devolver los datos parciales si fueran útiles para otros propósitos
        # Pero para renombrar, necesitamos todos.
        logger.warning(f"Método tradicional: Faltan datos críticos para renombrar. CUIT: {cuit}, Código: {codigo}, PV: {pv}, Nro: {nro}")
        # Devolver parciales para que extraer_datos_adicionales pueda usarlos
        return {
            'cuit': cuit, 'codigo': codigo, 'pv': pv, 'nro': nro,
            'cuil_afiliado': cuil_afiliado, 'nombre_afiliado': nombre_afiliado,
            'nombre_archivo': None # Indicar que no se pudo generar nombre completo
        }


    return {
        'cuit': cuit,
        'codigo': codigo,
        'pv': pv,
        'nro': nro,
        'cuil_afiliado': cuil_afiliado,
        'nombre_afiliado': nombre_afiliado,
        'nombre_archivo': f"{cuit}_{codigo}_{pv}_{nro}.pdf"
    } #

def leer_pagina(pdf_bytes_io, pagina_idx): # pdf_bytes_io es un io.BytesIO
    """Lee una página de PDF y extrae su texto."""
    try:
        # PdfReader necesita que el stream esté al inicio si se va a leer múltiples veces
        pdf_bytes_io.seek(0) 
        reader = PdfReader(pdf_bytes_io)
        if pagina_idx >= len(reader.pages):
            logger.warning(f"Se solicitó la página {pagina_idx+1} pero el PDF solo tiene {len(reader.pages)} páginas.")
            return ""
        texto_pagina = reader.pages[pagina_idx].extract_text()
        return texto_pagina if texto_pagina else ""
    except Exception as e:
        logger.error(f"Error al leer la página {pagina_idx+1} del PDF: {e}")
        return ""

def leer_pagina_ocr(pdf_bytes_io, pagina_idx): # pdf_bytes_io es un io.BytesIO
    """Aplica OCR a una página de PDF para extraer texto."""
    try:
        # convert_from_bytes necesita bytes, no un stream de BytesIO directamente en algunas versiones
        # Pero usualmente funciona con getvalue()
        pdf_bytes_io.seek(0)
        images = convert_from_bytes(pdf_bytes_io.getvalue(), first_page=pagina_idx+1, last_page=pagina_idx+1, dpi=300) # Aumentar DPI
        if images:
            return pytesseract.image_to_string(images[0], lang='spa') # Especificar idioma
        logger.warning(f"OCR no pudo generar imagen para la página {pagina_idx+1}")
        return ""
    except Exception as e:
        logger.error(f"Error en OCR para la página {pagina_idx+1}: {e}")
        return ""

def extraer_desde_dos_paginas(fh_pdf_bytes_io, file_name_for_log="archivo", use_ai=True): # fh es io.BytesIO
   """Extrae datos desde las dos primeras páginas del PDF usando OCR y IA."""
   texto_p1_raw = leer_pagina(fh_pdf_bytes_io, 0)
   texto_p2_raw = leer_pagina(fh_pdf_bytes_io, 1)
   
   texto_p1_ocr = ""
   texto_p2_ocr = ""

   # Aplicar OCR si el texto extraído es muy corto, o como opción adicional
   if len(texto_p1_raw.strip()) < 50: # Umbral bajo para forzar OCR si es casi vacío
       logger.info(f"Texto de página 1 (raw) muy corto para '{file_name_for_log}'. Intentando OCR...")
       texto_p1_ocr = leer_pagina_ocr(fh_pdf_bytes_io, 0)
   texto_final_p1 = texto_p1_ocr if texto_p1_ocr else texto_p1_raw

   if len(texto_p2_raw.strip()) < 50:
       logger.info(f"Texto de página 2 (raw) muy corto para '{file_name_for_log}'. Intentando OCR...")
       texto_p2_ocr = leer_pagina_ocr(fh_pdf_bytes_io, 1)
   texto_final_p2 = texto_p2_ocr if texto_p2_ocr else texto_p2_raw
   
   texto_completo_para_tradicional = (texto_final_p1 + "\n" + texto_final_p2).strip()

   logger.debug(f"--- TEXTO PÁGINA 1 ({file_name_for_log}) ---\n{texto_final_p1[:1000]}...")
   logger.debug(f"--- TEXTO PÁGINA 2 ({file_name_for_log}) ---\n{texto_final_p2[:1000]}...")
      
   # Intentar con IA PRIMERO si está habilitada
   if use_ai and factura_processor:
       logger.info(f"Intentando extraer con IA para '{file_name_for_log}'...")
       try:
           fh_pdf_bytes_io.seek(0) # Resetear stream para convert_from_bytes
           images_from_pdf = convert_from_bytes(fh_pdf_bytes_io.getvalue(), first_page=1, last_page=1, dpi=300) # Usar DPI más alto
           
           if images_from_pdf:
               image_pil = images_from_pdf[0]
               
               # Aquí no se llama a is_valid_factura, se va directo a extraer entidades
               # Si se quisiera validar primero, se llamaría a factura_processor.is_valid_factura(image_pil)
               
               entities_ia = factura_processor.extract_entities(image_pil)
               logger.info(f"Entidades extraídas por IA para '{file_name_for_log}': {entities_ia}")
               
               # Verificar confianza de predicciones
               es_confiable_ia, mensaje_confianza_ia = factura_processor.verificar_predicciones(entities_ia)
               logger.info(f"Evaluación del modelo IA para '{file_name_for_log}': {mensaje_confianza_ia}")
               
               if es_confiable_ia and \
                  all(k in entities_ia and entities_ia[k] for k in ['CUIT_PRESTADOR', 'PUNTO_VENTA', 'NUMERO_FACTURA']): # Asegurar que no estén vacíos
                   
                   cuit_ia = str(entities_ia['CUIT_PRESTADOR']).replace("-", "").replace(" ", "")
                   
                   tipo_ia_str = str(entities_ia.get('TIPO_FACTURA', 'FACTURA')).upper()
                   letra_ia_str = str(entities_ia.get('LETRA_FACTURA', 'C')).upper() # Default a C si no está
                   
                   codigo_ia = tipo_y_letra_a_codigo(tipo_ia_str, letra_ia_str)
                   
                   if not codigo_ia or codigo_ia == 0: # Si es 0 o None
                       # Fallback más inteligente para el código si no se detectó bien
                       logger.warning(f"IA: Código de comprobante no determinado o inválido para {tipo_ia_str} {letra_ia_str}. Intentando inferir.")
                       # Podríamos intentar inferir basado en otras claves o asumir un default más común.
                       # Por ahora, si no es válido, el método tradicional podría ser mejor.
                       # O se podría forzar un default como "05" para Factura C si es el caso más común.
                       if tipo_ia_str == "FACTURA" and letra_ia_str == "C": codigo_ia = "05"
                       elif tipo_ia_str == "FACTURA" and letra_ia_str == "B": codigo_ia = "03"
                       else: codigo_ia = "05" # Default general
                       logger.info(f"IA: Usando código de comprobante inferido/default: {codigo_ia}")
                   else:
                       codigo_ia = str(codigo_ia).zfill(2) # Asegurar 2 dígitos si es válido


                   pv_ia = limpiar_numero(entities_ia['PUNTO_VENTA'])
                   nro_ia = limpiar_numero(entities_ia['NUMERO_FACTURA'])
                   
                   if len(cuit_ia) == 11 and pv_ia.isdigit() and nro_ia.isdigit():
                       datos_ia_construidos = {
                           'cuit': cuit_ia,
                           'codigo': codigo_ia,
                           'pv': pv_ia,
                           'nro': nro_ia,
                           'nombre_archivo': f"{cuit_ia}_{codigo_ia}_{pv_ia}_{nro_ia}.pdf",
                           'cuil_afiliado': str(entities_ia.get('CUIT_AFILIADO', "")).replace("-", "").replace(" ", ""),
                           'nombre_afiliado': entities_ia.get('NOMBRE_AFILIADO', ""),
                           'dni_afiliado': str(entities_ia.get('DNI_AFILIADO', "")).replace(" ",""),
                           'texto_completo': texto_completo_para_tradicional, # Usar el texto de OCR/raw para datos adicionales
                       }
                       registrar_resultados_modelo(entities_ia, "IA_PDF", texto_completo_para_tradicional)
                       logger.info(f"IA extrajo datos válidos para '{file_name_for_log}': {datos_ia_construidos['nombre_archivo']}")
                       return datos_ia_construidos, None
                   else:
                       logger.warning(f"IA extrajo datos pero no son válidos para renombrar: CUIT:{cuit_ia}, PV:{pv_ia}, NRO:{nro_ia}")
               else:
                   logger.warning(f"IA no pudo extraer datos suficientes o confiables para '{file_name_for_log}': {mensaje_confianza_ia}")
           else: # if not images_from_pdf
               logger.warning(f"No se pudo convertir el PDF '{file_name_for_log}' a imagen para la IA.")
       except Exception as e:
           logger.error(f"Error en procesamiento con IA para '{file_name_for_log}': {e}", exc_info=True) # Añadido exc_info para más detalle
   
   # Si la IA falló o no está habilitada, volver al método tradicional
   logger.info(f"Usando método tradicional para '{file_name_for_log}' (o como respaldo)...")
   
   if not texto_completo_para_tradicional.strip(): # Si no hay nada de texto
       logger.error(f"No se pudo extraer texto (raw ni OCR) de '{file_name_for_log}'. No se puede procesar.")
       return None, f"❌ No se pudo extraer texto de '{file_name_for_log}'"

   datos_tradicionales = extraer_datos_flexible(texto_completo_para_tradicional)
   
   if datos_tradicionales and datos_tradicionales.get('nombre_archivo'):
       datos_tradicionales['texto_completo'] = texto_completo_para_tradicional
       # Registrar que se usó el método tradicional
       entities_dummy_trad = {k: v for k, v in datos_tradicionales.items() if k not in ['texto_completo', 'nombre_archivo']}
       registrar_resultados_modelo(entities_dummy_trad, "Tradicional_PDF", texto_completo_para_tradicional)
       logger.info(f"Método tradicional extrajo datos para '{file_name_for_log}': {datos_tradicionales['nombre_archivo']}")
       return datos_tradicionales, None
   else:
       error_msg = f"❌ Método tradicional no pudo extraer datos de '{file_name_for_log}'"
       if datos_tradicionales: # Si devolvió datos parciales pero sin nombre_archivo
           logger.warning(f"Método tradicional extrajo datos parciales pero insuficientes para renombrar '{file_name_for_log}'. Datos: {datos_tradicionales}")
           # Registrar intento aunque sea parcial
           entities_dummy_trad_parcial = {k: v for k, v in datos_tradicionales.items() if k not in ['texto_completo', 'nombre_archivo']}
           registrar_resultados_modelo(entities_dummy_trad_parcial, "Tradicional_PDF_Parcial", texto_completo_para_tradicional)

       logger.error(error_msg)
       return None, error_msg


def extraer_desde_imagen(image_bytes, image_name_for_log="imagen", use_ai=True): # image_bytes son los bytes de la imagen
   """Extrae datos desde una imagen."""
   try:
       image_pil = Image.open(io.BytesIO(image_bytes))
       texto_ocr_imagen = pytesseract.image_to_string(image_pil, lang='spa') # Especificar idioma
       logger.debug(f"Texto extraído por OCR de '{image_name_for_log}':\n{texto_ocr_imagen[:500]}...")
   except Exception as e:
       logger.error(f"Error al abrir o hacer OCR a la imagen '{image_name_for_log}': {e}")
       return None, f"Error de OCR en imagen '{image_name_for_log}'"

   if not texto_ocr_imagen.strip():
        logger.warning(f"OCR no extrajo texto de la imagen '{image_name_for_log}'.")
        # No intentar IA si no hay texto, ya que nuestro LayoutLM se basa en OCR externo
        return None, f"OCR no extrajo texto de imagen '{image_name_for_log}'"

   # Intentar con IA si está habilitada (requiere que la IA pueda procesar la imagen directamente)
   if use_ai and factura_processor:
       logger.info(f"Intentando extraer con IA desde imagen '{image_name_for_log}'...")
       try:
           entities_ia_img = factura_processor.extract_entities(image_pil) # Pasar la imagen PIL
           logger.info(f"IA extrajo entidades de imagen '{image_name_for_log}': {entities_ia_img}")
           
           es_confiable_ia_img, mensaje_confianza_ia_img = factura_processor.verificar_predicciones(entities_ia_img)
           logger.info(f"Evaluación del modelo IA en imagen '{image_name_for_log}': {mensaje_confianza_ia_img}")
           
           if es_confiable_ia_img and \
              all(k in entities_ia_img and entities_ia_img[k] for k in ['CUIT_PRESTADOR', 'PUNTO_VENTA', 'NUMERO_FACTURA']):

               cuit_ia_img = str(entities_ia_img['CUIT_PRESTADOR']).replace("-", "").replace(" ", "")
               tipo_ia_img_str = str(entities_ia_img.get('TIPO_FACTURA', 'FACTURA')).upper()
               letra_ia_img_str = str(entities_ia_img.get('LETRA_FACTURA', 'C')).upper()
               codigo_ia_img = tipo_y_letra_a_codigo(tipo_ia_img_str, letra_ia_img_str)
               if not codigo_ia_img or codigo_ia_img == 0: 
                   codigo_ia_img = "05" # Default
                   logger.info(f"IA (imagen): Usando código de comprobante default: {codigo_ia_img}")
               else:
                   codigo_ia_img = str(codigo_ia_img).zfill(2)


               pv_ia_img = limpiar_numero(entities_ia_img['PUNTO_VENTA'])
               nro_ia_img = limpiar_numero(entities_ia_img['NUMERO_FACTURA'])

               if len(cuit_ia_img) == 11 and pv_ia_img.isdigit() and nro_ia_img.isdigit():
                   datos_ia_img_construidos = {
                       'cuit': cuit_ia_img, 'codigo': codigo_ia_img, 'pv': pv_ia_img, 'nro': nro_ia_img,
                       'nombre_archivo': f"{cuit_ia_img}_{codigo_ia_img}_{pv_ia_img}_{nro_ia_img}.pdf", # Asumimos que se renombrará como PDF
                       'cuil_afiliado': str(entities_ia_img.get('CUIT_AFILIADO', "")).replace("-", "").replace(" ", ""),
                       'nombre_afiliado': entities_ia_img.get('NOMBRE_AFILIADO', ""),
                       'dni_afiliado': str(entities_ia_img.get('DNI_AFILIADO', "")).replace(" ",""),
                       'texto_completo': texto_ocr_imagen,
                   }
                   registrar_resultados_modelo(entities_ia_img, "IA_Imagen", texto_ocr_imagen)
                   logger.info(f"IA extrajo datos válidos de la imagen '{image_name_for_log}': {datos_ia_img_construidos['nombre_archivo']}")
                   return datos_ia_img_construidos, None
               else:
                   logger.warning(f"IA (imagen) '{image_name_for_log}' extrajo datos pero no son válidos para renombrar.")
           else:
                logger.warning(f"IA (imagen) '{image_name_for_log}' no pudo extraer datos suficientes o confiables: {mensaje_confianza_ia_img}")
       except Exception as e:
           logger.error(f"Error en procesamiento de imagen '{image_name_for_log}' con IA: {e}", exc_info=True)
   
   # Si la IA falló o no está habilitada, usar método tradicional con el texto OCR de la imagen
   logger.info(f"Usando método tradicional para imagen '{image_name_for_log}' (o como respaldo)...")
   datos_trad_img = extraer_datos_flexible(texto_ocr_imagen)
   if datos_trad_img and datos_trad_img.get('nombre_archivo'):
       datos_trad_img['texto_completo'] = texto_ocr_imagen
       entities_dummy_trad_img = {k:v for k,v in datos_trad_img.items() if k not in ['texto_completo', 'nombre_archivo']}
       registrar_resultados_modelo(entities_dummy_trad_img, "Tradicional_Imagen", texto_ocr_imagen)
       logger.info(f"Método tradicional extrajo datos de imagen '{image_name_for_log}': {datos_trad_img['nombre_archivo']}")
       return datos_trad_img, None
   else:
       error_msg_img = f"❌ Método tradicional no pudo extraer datos de imagen '{image_name_for_log}'"
       if datos_trad_img:
            logger.warning(f"Método tradicional (imagen) '{image_name_for_log}' extrajo datos parciales pero insuficientes para renombrar. Datos: {datos_trad_img}")
            entities_dummy_trad_img_parcial = {k:v for k,v in datos_trad_img.items() if k not in ['texto_completo', 'nombre_archivo']}
            registrar_resultados_modelo(entities_dummy_trad_img_parcial, "Tradicional_Imagen_Parcial", texto_ocr_imagen)
       logger.error(error_msg_img)
       return None, error_msg_img


def extraer_datos_desde_nombre(nombre):
   """Extrae datos desde el nombre del archivo si ya está renombrado."""
   match = re.match(PATTERN_RENAMED, nombre) # Usar el patrón global
   if match:
       cuit, codigo_str, pv, nro = match.groups()
       try:
           codigo_int = int(codigo_str) # El código debería ser un entero para consistencia
           return {
               'cuit': cuit,
               'codigo': str(codigo_int).zfill(2), # Guardar como string de 2 dígitos
               'pv': pv, # Ya deberían estar limpios por el patrón
               'nro': nro, # Ya deberían estar limpios por el patrón
               'nombre_archivo': nombre
           }
       except ValueError:
            logger.error(f"Error al convertir código '{codigo_str}' a entero en nombre de archivo: {nombre}")
            return None
   return None

# ========== FUNCIONES PARA GENERACIÓN DE TXT ========== (Se mantienen, pero se revisan llamadas)
def formatear_fecha(fecha_str_input):
    """Convierte una fecha a formato DD/MM/AAAA, manejando varios formatos de entrada."""
    if not fecha_str_input: return ""
    fecha_str = str(fecha_str_input).strip().split(" ")[0] # Tomar solo la parte de la fecha si hay hora

    # Formatos a intentar, del más específico al más general
    formatos_entrada = [
        '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', 
        '%Y/%m/%d', '%d.%m.%Y', '%d/%m/%y', '%d-%m-%y'
    ]
    for fmt in formatos_entrada:
        try:
            dt_obj = datetime.strptime(fecha_str, fmt)
            # Si el año es de 2 dígitos, convertir a 4 (asumiendo siglo XXI)
            if dt_obj.year < 100: # Ej. 25 se convierte a 2025
                dt_obj = dt_obj.replace(year=dt_obj.year + 2000)
            return dt_obj.strftime('%d/%m/%Y')
        except ValueError:
            continue
    
    # Si ninguno de los formatos estándar funciona, intentar regex más permisivas
    # Para DD/MM/AAAA o D/M/AAAA
    match_slash = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', fecha_str)
    if match_slash:
        d, m, y = match_slash.groups()
        y = '20' + y if len(y) == 2 else y
        try: # Verificar validez de la fecha construida
            return datetime(int(y), int(m), int(d)).strftime('%d/%m/%Y')
        except ValueError:
            pass # Continuar si no es válida

    # Para AAAA-MM-DD o AAAA/MM/DD
    match_iso_like = re.match(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', fecha_str)
    if match_iso_like:
        y, m, d = match_iso_like.groups()
        try:
            return datetime(int(y), int(m), int(d)).strftime('%d/%m/%Y')
        except ValueError:
            pass
            
    logger.warning(f"Formato de fecha no reconocido, devolviendo original: '{fecha_str_input}'")
    return str(fecha_str_input).strip() # Devolver original si todo falla


def extraer_fecha(texto): # Para fecha de comprobante
    """Extrae la fecha de emisión del documento."""
    # Prioridad: "Fecha de Emisión: DD/MM/AAAA"
    match_emision = re.search(r"Fecha\s*(?:de)?\s*Emisi[oó]n:?\s*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})", texto, re.I)
    if match_emision:
        return formatear_fecha(match_emision.group(1))

    # Segundo: "Fecha: DD/MM/AAAA" (si no es "Fecha de Vencimiento")
    # Usar lookbehind negativo para evitar "Vencimiento" o "Vto"
    match_fecha_simple = re.search(r"(?<!Vencimiento\s)(?<!Vto\.\s)Fecha:?\s*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})", texto, re.I)
    if match_fecha_simple:
        return formatear_fecha(match_fecha_simple.group(1))

    # Tercero: buscar fechas genéricas DD/MM/AAAA o AAAA-MM-DD
    # Es riesgoso tomar la primera que aparezca, pero como fallback
    # Intentar buscar cerca de palabras clave como "Factura", "Comprobante" o el número de comprobante
    
    # Buscar cualquier fecha en formato DD/MM/AAAA o D/M/AA o D.M.AAAA, etc.
    # Esta regex es más general.
    fechas_genericas = re.findall(r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b", texto)
    if fechas_genericas:
        # Intentar dar prioridad a fechas que no parezcan de vencimiento
        fechas_validas_formateadas = []
        for f_str in fechas_genericas:
            # Evitar fechas que claramente son de vencimiento si están cerca de "Vto." o "Vencimiento"
            idx_f = texto.find(f_str)
            texto_previo = texto[max(0, idx_f-30):idx_f].lower() # 30 chars antes
            if "vencimiento" not in texto_previo and "vto" not in texto_previo:
                formateada = formatear_fecha(f_str)
                if len(formateada) == 10: # "DD/MM/AAAA"
                     fechas_validas_formateadas.append(formateada)
        
        if fechas_validas_formateadas:
            # Podríamos tener una lógica para elegir la "mejor" si hay varias
            # Por ahora, tomamos la primera válida encontrada
            return fechas_validas_formateadas[0]
            
    logger.warning("No se pudo extraer fecha de comprobante con certeza, usando fecha actual.")
    return datetime.now().strftime("%d/%m/%Y")


def extraer_cae(texto):
    """Extrae el CAE/CAI y el tipo de emisión (E/I)."""
    # Patrones para CAE (electrónico)
    patrones_cae_e = [
        r"CAE\s*N[°º]?:?\s*(\d{14})",
        r"C\.?A\.?E\.?[:\s#]+(\d{14})",
        r"Comprobante Autorizado\s*.*?(\d{14})", # A veces CAE está implícito
    ]
    for patron in patrones_cae_e:
        match = re.search(patron, texto, re.I | re.DOTALL) # DOTALL para multilínea
        if match:
            return match.group(1), "E"

    # Patrones para CAI (impreso, menos común ahora)
    patrones_cai_i = [
        r"CAI\s*N[°º]?:?\s*(\d{14})"
    ]
    for patron in patrones_cai_i:
        match = re.search(patron, texto, re.I)
        if match:
            return match.group(1), "I"
            
    # Fallback: cualquier número de 14 dígitos que parezca un CAE
    # Buscar cerca de "Vencimiento CAE" o al final de la factura
    match_cae_vto = re.search(r"(\d{14})\s*Vto\.?\s*CAE", texto, re.I)
    if match_cae_vto:
        return match_cae_vto.group(1), "E"

    # Último recurso: un número de 14 dígitos solo
    # Esto es muy propenso a errores, pero se puede intentar si no hay más
    # match_14_digitos = re.search(r"\b(\d{14})\b", texto)
    # if match_14_digitos:
    #     # Verificar si está cerca del final o de palabras clave
    #     logger.warning("CAE encontrado como número de 14 dígitos aislado, podría ser incorrecto.")
    #     return match_14_digitos.group(1), "E"
        
    return None, None


def extraer_importe(texto):
    """Extrae el importe total del documento."""
    # Patrones más comunes y específicos primero
    patrones = [
        r"TOTAL\s*\$?\s*([\d.,]+)", # TOTAL $ 123.456,78 o TOTAL 123456,78
        r"Importe\s*Total\s*\$?\s*([\d.,]+)",
        r"TOTAL\s+A\s+PAGAR\s*\$?\s*([\d.,]+)",
        r"A\s+PAGAR\s*\$?\s*([\d.,]+)",
        r"Paga\s+con\s*\$?\s*([\d.,]+)" # A veces para recibos
    ]
    
    texto_sin_espacios_internos_numeros = re.sub(r'(\d)\s+(\d)', r'\1\2', texto) # Unir números separados por espacios: "1 23" -> "123"

    for patron in patrones:
        match = re.search(patron, texto_sin_espacios_internos_numeros, re.I | re.DOTALL)
        if match:
            importe_str = match.group(1)
            # Limpieza robusta: quitar todo excepto dígitos y el último separador decimal (coma o punto)
            # Primero, quitar separadores de miles (puntos si el decimal es coma, o comas si el decimal es punto)
            if ',' in importe_str and '.' in importe_str:
                if importe_str.rfind('.') > importe_str.rfind(','): # Formato 1,234.56
                    importe_str_limpio = importe_str.replace(',', '')
                else: # Formato 1.234,56
                    importe_str_limpio = importe_str.replace('.', '')
            elif '.' in importe_str: # Formato 1234.56 (o 1.234 si no hay decimales)
                partes = importe_str.split('.')
                if len(partes[-1]) == 2 : # Es decimal
                     importe_str_limpio = "".join(partes[:-1]) + partes[-1]
                else: # No hay decimales, el punto es separador de miles
                     importe_str_limpio = "".join(partes)

            elif ',' in importe_str: # Formato 1234,56 (o 1,234 si no hay decimales)
                partes = importe_str.split(',')
                if len(partes[-1]) == 2 : # Es decimal
                     importe_str_limpio = "".join(partes[:-1]) + partes[-1]
                else: # No hay decimales, la coma es separador de miles
                     importe_str_limpio = "".join(partes)
            else: # Sin puntos ni comas
                importe_str_limpio = importe_str

            importe_final_digitos = ''.join(c for c in importe_str_limpio if c.isdigit())
            if importe_final_digitos:
                return importe_final_digitos # Devolver solo dígitos "centavos incluidos"
    
    # Fallback: buscar el número más grande con formato de moneda si no hay etiquetas claras
    # Esto es más arriesgado.
    logger.warning("No se encontró importe con etiqueta clara.")
    return None


def extraer_periodo(texto):
    """Extrae el período facturado en formato MMAAAA."""
    # "Período Facturado Desde: dd/mm/yyyy Hasta: dd/mm/yyyy" -> tomar mes y año del "Desde"
    match_desde_hasta = re.search(r"Per[ií]odo\s*(?:Facturado)?\s*Desde:?\s*\d{1,2}/(\d{1,2})/(\d{4})", texto, re.I)
    if match_desde_hasta:
        mes = match_desde_hasta.group(1).zfill(2)
        anio = match_desde_hasta.group(2)
        return f"{mes}{anio}"

    # "Período: MM/AAAA" o "Periodo Correspondiente a: MM/AAAA"
    match_periodo_directo = re.search(r"Per[ií]odo(?: Correspondiente a)?:?\s*(\d{1,2})/(\d{4})", texto, re.I)
    if match_periodo_directo:
        mes = match_periodo_directo.group(1).zfill(2)
        anio = match_periodo_directo.group(2)
        return f"{mes}{anio}"

    # "mes de XXX de YYYY"
    meses_es = {
        "enero": "01", "febrero": "02", "marzo": "03", "abril": "04", "mayo": "05", "junio": "06",
        "julio": "07", "agosto": "08", "septiembre": "09", "setiembre": "09", 
        "octubre": "10", "noviembre": "11", "diciembre": "12"
    }
    match_mes_literal = re.search(r"mes\s+de\s+(" + "|".join(meses_es.keys()) + r")\s*(?:de\s*)?(\d{4})", texto, re.I)
    if match_mes_literal:
        mes_nombre = match_mes_literal.group(1).lower()
        anio = match_mes_literal.group(2)
        if mes_nombre in meses_es:
            return f"{meses_es[mes_nombre]}{anio}"
            
    # Si no se encuentra, usar la fecha de emisión del comprobante para inferir el período
    # (esto asume que el período es el mes de la factura)
    fecha_emision_str = extraer_fecha(texto) # Usar la función de extraer fecha de comprobante
    if fecha_emision_str and len(fecha_emision_str) == 10: # DD/MM/AAAA
        try:
            dt_emision = datetime.strptime(fecha_emision_str, "%d/%m/%Y")
            logger.info(f"Usando mes/año de fecha de emisión ({fecha_emision_str}) como período.")
            return dt_emision.strftime("%m%Y")
        except ValueError:
            pass

    logger.warning("No se pudo determinar el período, usando mes/año actual.")
    ahora = datetime.now()
    return f"{ahora.month:02d}{ahora.year}"


def map_actividad(texto_factura_lower):
    """Determina el código de actividad y la bandera de dependencia."""
    # Transporte
    if "transporte" in texto_factura_lower or "traslado" in texto_factura_lower or " remise" in texto_factura_lower or re.search(r"\bkm\b", texto_factura_lower):
        dep_flag = "S" if any(term in texto_factura_lower for term in ["discapacidad", "discapac."]) else "N"
        return "096", dep_flag
    
    # Prestaciones profesionales
    profesionales_terminos = [
        "psicolog", "musicoterap", "kinesiolog", "fonoaudiolog", "psicopedagog", "terapia ocupacional"
    ]
    if any(term in texto_factura_lower for term in profesionales_terminos):
        return "091", "N" # Asumimos N para estas por defecto
    
    # Estimulación temprana
    if "estimulaci[oó]n temprana" in texto_factura_lower:
        return "085", "N"
        
    # Apoyo a la integración escolar / Maestra Integradora
    if any(term in texto_factura_lower for term in ["m[oó]dulo de apoyo", "apoyo a la integraci[oó]n", "maestra integradora"]):
        return "089", "N"
        
    # Centro de día / Centro educativo terapéutico
    if "centro de d[ií]a" in texto_factura_lower or "centro educativo terap[eé]utico" in texto_factura_lower:
        return "083", "N" # Ejemplo, verificar código correcto

    # Si contiene "sesiones", "honorarios profesionales", "tratamiento" y no es ninguna de las anteriores:
    if any(term in texto_factura_lower for term in ["sesiones", "honorarios profesionales", "tratamiento", "terapia"]):
        return "090", "N" # Código genérico para terapias
        
    # Por defecto
    dep_flag = "S" if any(term in texto_factura_lower for term in ["discapacidad", "discapac."]) else "N"
    logger.info(f"Actividad no claramente identificada, usando default 090 y dependencia: {dep_flag}")
    return "090", dep_flag


def cantidad_por_actividad(cod_actividad):
    """Devuelve la cantidad predeterminada según el código de actividad."""
    if cod_actividad == "096":  # Transporte
        return "001500" # Ejemplo: 1500 km o unidades, verificar requerimiento
    if cod_actividad in {"090", "091", "085"}:  # Terapias y profesionales, estimulación
        # Podría ser número de sesiones. Si se factura mensual, puede ser 1.
        # Si es por sesión, y se facturan varias, este valor debería ser el total de sesiones.
        # Por ahora, un default conservador.
        return "000001" # Asumiendo 1 "servicio" o "mes"
    if cod_actividad in {"089", "083"}: # Apoyo integración, Centro de día (usualmente mensual)
        return "000001"
    return "000001" # Default genérico

def extraer_datos_adicionales(datos_renombrado, use_ai=True): # datos_renombrado es el dict de extraer_datos_flexible o de IA
    """Extrae datos adicionales para la generación del TXT, priorizando IA si está disponible y es confiable."""
    
    # Texto completo de la factura (obtenido por OCR/raw text en pasos previos)
    texto_factura_completo = datos_renombrado.get('texto_completo', '')
    if not texto_factura_completo:
        logger.warning(f"No hay texto completo en datos_renombrado para {datos_renombrado.get('nombre_archivo', 'archivo desconocido')}. No se pueden extraer datos adicionales.")
        return None
    
    texto_factura_lower = texto_factura_completo.lower() # Para búsquedas insensibles a mayúsculas

    # Datos básicos ya existentes del renombrado (o de la IA si ya se usó para renombrar)
    cuit_prestador = datos_renombrado.get('cuit')
    codigo_comprobante = str(datos_renombrado.get('codigo', "")).zfill(2)
    punto_venta = datos_renombrado.get('pv')
    numero_comprobante = datos_renombrado.get('nro')

    # Variables para almacenar resultados de IA y tradicionales
    fecha_cbte_ia = None
    nro_cae_ia = None
    tipo_emision_ia = None
    importe_ia = None
    periodo_ia = None
    actividad_texto_ia = None
    
    # --- INICIALIZACIÓN CORRECTA DE VARIABLES PARA DATOS DEL AFILIADO DESDE IA ---
    # Primero, tomar los valores que podrían ya existir en datos_renombrado (si la IA ya corrió para renombrar)
    # o si fueron extraídos por métodos tradicionales y pasados.
    cuil_afiliado_final = datos_renombrado.get('cuil_afiliado') 
    nombre_afiliado_final = datos_renombrado.get('nombre_afiliado')
    dni_afiliado_final = datos_renombrado.get('dni_afiliado') # Obtener DNI si ya existe en datos_renombrado
    # --- FIN DE INICIALIZACIÓN ---

    # Si se debe usar IA y el procesador está listo
    if use_ai and factura_processor:
        file_id = datos_renombrado.get('file_id')
        nombre_archivo_log = datos_renombrado.get('nombre_archivo') or datos_renombrado.get('nombre_original', 'archivo desconocido')
        
        if file_id:
            logger.info(f"Intentando (re)extraer con IA para datos adicionales de '{nombre_archivo_log}' (ID: {file_id})")
            try:
                service = autenticar() 
                request = service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status: logger.debug(f"Descargando para IA en extraer_datos_adicionales: {int(status.progress() * 100)}%")
                fh.seek(0)
                
                images = convert_from_bytes(fh.getvalue(), first_page=1, last_page=1, dpi=300)
                if images:
                    image = images[0]
                    entities_ia_adicionales = factura_processor.extract_entities(image)
                    logger.info(f"IA (datos adicionales) para '{nombre_archivo_log}' extrajo: {entities_ia_adicionales}")

                    # Actualizar/sobrescribir con datos de IA si se encontraron y son válidos
                    if entities_ia_adicionales.get('FECHA_EMISION'):
                        fecha_cbte_ia = formatear_fecha(entities_ia_adicionales['FECHA_EMISION'])
                    if entities_ia_adicionales.get('CAE'):
                        nro_cae_ia = entities_ia_adicionales['CAE']
                        tipo_emision_ia = "E" 
                    if entities_ia_adicionales.get('IMPORTE'):
                        importe_ia_str = str(entities_ia_adicionales['IMPORTE'])
                        importe_ia = ''.join(c for c in importe_ia_str if c.isdigit()) 
                    if entities_ia_adicionales.get('PERIODO'):
                        periodo_str_ia = str(entities_ia_adicionales['PERIODO'])
                        match_p = re.search(r"(\d{1,2})[/-]?(\d{4})", periodo_str_ia)
                        if match_p: periodo_ia = f"{match_p.group(1).zfill(2)}{match_p.group(2)}"
                        elif len(periodo_str_ia.replace(" ","")) == 6 and periodo_str_ia.replace(" ","").isdigit():
                            periodo_ia = periodo_str_ia.replace(" ","")

                    if entities_ia_adicionales.get('ACTIVIDAD'):
                        actividad_texto_ia = entities_ia_adicionales['ACTIVIDAD']
                    
                    # Actualizar datos del afiliado si la IA los mejora o encuentra
                    if entities_ia_adicionales.get('CUIT_AFILIADO'): # Si la IA da CUIT_AFILIADO, usarlo
                        cuil_afiliado_final = str(entities_ia_adicionales['CUIT_AFILIADO']).replace("-", "").replace(" ", "")
                    if entities_ia_adicionales.get('NOMBRE_AFILIADO'): # Si la IA da NOMBRE_AFILIADO, usarlo
                        nombre_afiliado_final = entities_ia_adicionales['NOMBRE_AFILIADO']
                    if entities_ia_adicionales.get('DNI_AFILIADO'): # Si la IA da DNI_AFILIADO, usarlo
                        dni_afiliado_final = str(entities_ia_adicionales['DNI_AFILIADO']).replace(" ","").replace(".","") # Limpiar puntos también

            except Exception as e_ia_ad:
                logger.error(f"Error al (re)extraer con IA para datos adicionales de '{nombre_archivo_log}': {e_ia_ad}", exc_info=True)
        else: # if not file_id (no se puede descargar para IA)
             logger.info(f"No se usará IA en extraer_datos_adicionales para '{nombre_archivo_log}' (no hay file_id para descargar imagen).")


    # Lógica de asignación final: priorizar IA, luego datos_renombrado (que podría tener datos de una IA previa), luego tradicional
    fecha_comprobante = fecha_cbte_ia or datos_renombrado.get('IA_FECHA_EMISION') or datos_renombrado.get('fecha_cbte') # Asumiendo que 'fecha_cbte' podría estar en datos_renombrado
    if not fecha_comprobante: fecha_comprobante = extraer_fecha(texto_factura_completo)
    fecha_comprobante = formatear_fecha(fecha_comprobante)
    
    if nro_cae_ia:
        numero_cae = nro_cae_ia
        tipo_emision_cae = tipo_emision_ia or "E"
    elif datos_renombrado.get('IA_CAE'):
        numero_cae = datos_renombrado.get('IA_CAE')
        tipo_emision_cae = "E"
    elif datos_renombrado.get('nro_cae'): # Si ya existía un CAE no-IA
        numero_cae = datos_renombrado.get('nro_cae')
        tipo_emision_cae = datos_renombrado.get('tipo_emision', "E")
    else:
        numero_cae, tipo_emision_cae = extraer_cae(texto_factura_completo)
    
    if importe_ia:
        importe_total = importe_ia
    elif datos_renombrado.get('IA_IMPORTE'):
        importe_total = ''.join(c for c in str(datos_renombrado.get('IA_IMPORTE')) if c.isdigit())
    elif datos_renombrado.get('importe'):
        importe_total = ''.join(c for c in str(datos_renombrado.get('importe')) if c.isdigit())
    else:
        importe_total = extraer_importe(texto_factura_completo)

    if periodo_ia:
        periodo_facturado = periodo_ia
    elif datos_renombrado.get('IA_PERIODO'):
        periodo_facturado = datos_renombrado.get('IA_PERIODO') 
    elif datos_renombrado.get('periodo'):
        periodo_facturado = datos_renombrado.get('periodo')
    else:
        periodo_facturado = extraer_periodo(texto_factura_completo)

    if actividad_texto_ia:
        codigo_actividad, dependencia_flag = map_actividad(actividad_texto_ia.lower())
    elif datos_renombrado.get('IA_ACTIVIDAD'):
         codigo_actividad, dependencia_flag = map_actividad(datos_renombrado.get('IA_ACTIVIDAD').lower())
    elif datos_renombrado.get('actividad') and datos_renombrado.get('dep'): # Si ya existen de forma no-IA
        codigo_actividad = datos_renombrado.get('actividad')
        dependencia_flag = datos_renombrado.get('dep')
    else:
        codigo_actividad, dependencia_flag = map_actividad(texto_factura_lower)
    
    cantidad = cantidad_por_actividad(codigo_actividad)

    # Los datos del afiliado (cuil_afiliado_final, nombre_afiliado_final, dni_afiliado_final)
    # ya se inicializaron y se actualizaron si la IA encontró algo nuevo.

    if not (cuit_prestador and codigo_comprobante and punto_venta and numero_comprobante and codigo_comprobante != "00"):
        logger.error(f"Faltan datos básicos CUIT/Comp/PV/Nro en datos_renombrado para {datos_renombrado.get('nombre_archivo', 'archivo desconocido')}")
        return None

    return {
        'cuit_pre': cuit_prestador,
        'codigo_cbte': codigo_comprobante,
        'pv': punto_venta,
        'nro': numero_comprobante,
        'fecha_cbte': fecha_comprobante or datetime.now().strftime("%d/%m/%Y"), 
        'tipo_emision': tipo_emision_cae or "E", 
        'nro_cae': numero_cae or "00000000000000", 
        'importe': importe_total or "0", 
        'periodo': periodo_facturado or datetime.now().strftime("%m%Y"), 
        'actividad': codigo_actividad or "090", 
        'cantidad': cantidad, 
        'dep': dependencia_flag or "N", 
        'cuil_afiliado': cuil_afiliado_final, # Usar el valor final (puede ser de IA o del renombrado)
        'nombre_afiliado': nombre_afiliado_final, # Usar el valor final
        'dni_afiliado': dni_afiliado_final, # << AHORA ESTA VARIABLE ESTÁ DEFINIDA Y SE RETORNA
        'texto_factura': texto_factura_lower 
    }


def verificar_formato_txt(nombre_txt):
   """Verifica que el formato del archivo TXT sea correcto."""
   # Esta función parece correcta, la mantengo como estaba.
   try:
       with open(nombre_txt, 'r', encoding='utf-8') as f:
           lineas = f.readlines()
       
       errores = []
       if not lineas:
           errores.append("El archivo TXT está vacío.")

       for i, linea in enumerate(lineas, 1):
           campos = linea.strip().split('|')
           
           if len(campos) != 19:
               errores.append(f"Línea {i}: Número incorrecto de campos ({len(campos)}, debería ser 19). Contenido: '{linea.strip()}'")
               continue
           
           # Ejemplo de verificación: Fecha de comprobante (campo 10, índice 9)
           fecha_cbte_txt = campos[9]
           if not re.match(r'^\d{2}/\d{2}/\d{4}$', fecha_cbte_txt):
               errores.append(f"Línea {i}: Formato de fecha de comprobante incorrecto: '{fecha_cbte_txt}' (esperado DD/MM/AAAA).")

           # Ejemplo: Importe (campo 14, índice 13)
           importe_txt = campos[13]
           if not (len(importe_txt) == 14 and importe_txt.isdigit()):
                errores.append(f"Línea {i}: Formato de importe incorrecto: '{importe_txt}' (esperado 14 dígitos).")

           # CAE (campo 11, índice 10)
           nro_cae_txt = campos[10]
           if not (len(nro_cae_txt) == 14 and (nro_cae_txt.isdigit() or nro_cae_txt == "00000000000000")): # Permitir 14 ceros si no hay CAE
               errores.append(f"Línea {i}: Formato de CAE incorrecto: '{nro_cae_txt}' (esperado 14 dígitos o 14 ceros).")


       if errores:
           logger.warning("Se encontraron los siguientes errores de formato en el archivo TXT generado:")
           for error in errores:
               logger.warning(f"  - {error}")
           return False
       
       logger.info(f"Verificación del archivo TXT '{nombre_txt}': CORRECTA.")
       return True
   
   except Exception as e:
       logger.error(f"Error crítico al verificar archivo TXT '{nombre_txt}': {e}")
       return False


def construir_linea(rnos_obra_social, fila_excel_afiliado, info_factura_procesada):
    """Construye una línea del archivo TXT con el formato específico."""
    
    # Campos requeridos de info_factura_procesada con defaults más robustos
    campos_requeridos_factura = {
        "cuit_pre": "00000000000", "codigo_cbte": "00", "pv": "00000", "nro": "00000000",
        "fecha_cbte": datetime.now().strftime("%d/%m/%Y"), 
        "tipo_emision": "E", "nro_cae": "00000000000000",
        "importe": "0", # Se formateará a 14 dígitos más adelante
        "periodo": datetime.now().strftime("%m%Y"), 
        "actividad": "090", "cantidad": "000001", "dep": "N"
    }
    
    for campo, default_valor in campos_requeridos_factura.items():
        if campo not in info_factura_procesada or not info_factura_procesada[campo]:
            logger.warning(f"Dato faltante o vacío en info_factura_procesada para '{campo}', usando default: '{default_valor}' para CUIL {fila_excel_afiliado.get('cuil', 'Desconocido')}")
            info_factura_procesada[campo] = default_valor
            
    # Campos requeridos de fila_excel_afiliado
    cuil_afiliado = str(fila_excel_afiliado.get("cuil", "00000000000")).strip()
    codigo_cert_afiliado = str(fila_excel_afiliado.get("codigo_certificado", "")).strip()
    venc_cert_afiliado = str(fila_excel_afiliado.get("vencimiento_certificado", "")).strip() # Ya debería estar DD/MM/AAAA
    provincia_afiliado = str(fila_excel_afiliado.get("provincia", "00")).strip().zfill(2)

    # Formateo y validación final de los datos de la factura
    pv_fmt = str(info_factura_procesada["pv"]).zfill(5)
    nro_fmt = str(info_factura_procesada["nro"]).zfill(8)
    
    importe_str_raw = str(info_factura_procesada["importe"])
    importe_digitos = ''.join(c for c in importe_str_raw if c.isdigit())
    importe_fmt = importe_digitos.zfill(14)
    
    fecha_cbte_fmt = formatear_fecha(info_factura_procesada["fecha_cbte"]) # Re-formatear por si acaso
    nro_cae_fmt = str(info_factura_procesada["nro_cae"]).strip()
    if not (len(nro_cae_fmt) == 14 and nro_cae_fmt.isdigit()):
        nro_cae_fmt = "00000000000000" # Default si es inválido después de todo

    # Vencimiento certificado (ya debería estar formateado, pero asegurar)
    venc_cert_fmt = formatear_fecha(venc_cert_afiliado)
    if len(venc_cert_fmt) != 10: # Si el formateo falla
        logger.warning(f"Fecha de vencimiento de certificado inválida '{venc_cert_afiliado}' para CUIL {cuil_afiliado}, usando default.")
        venc_cert_fmt = "01/01/1900" # Un default muy obvio si falla

    # Código certificado (38 caracteres)
    codigo_cert_fmt = codigo_cert_afiliado.ljust(38, "0")[:38]


    linea_campos = [
       "DS",
       rnos_obra_social,
       cuil_afiliado,
       codigo_cert_fmt,
       venc_cert_fmt,
       info_factura_procesada["periodo"],
       info_factura_procesada["cuit_pre"],
       str(info_factura_procesada["codigo_cbte"]).zfill(2),
       info_factura_procesada["tipo_emision"],
       fecha_cbte_fmt,
       nro_cae_fmt,
       pv_fmt,
       nro_fmt,
       importe_fmt,
       importe_fmt, # Duplicado
       str(info_factura_procesada["actividad"]).zfill(3),
       str(info_factura_procesada["cantidad"]).zfill(6),
       provincia_afiliado,
       info_factura_procesada["dep"],
    ]
    
    # Verificar que todos los campos sean strings antes de unir
    for i, campo_val in enumerate(linea_campos):
        if not isinstance(campo_val, str):
            logger.error(f"Error crítico: Campo en la línea TXT no es string en índice {i}, valor: {campo_val}, CUIL: {cuil_afiliado}")
            # Esto podría indicar un error en la lógica de defaults o extracción.
            # Intentar convertir a string como último recurso para no detener todo.
            linea_campos[i] = str(campo_val)


    return "|".join(linea_campos)


def subir_log(service, filepath, folder_id_drive): # Cambiado nombre de folder_id
   """Sube un archivo de log a Google Drive."""
   try:
       nombre_archivo_log = os.path.basename(filepath)
       logger.info(f"Subiendo log '{nombre_archivo_log}' a Google Drive folder ID: {folder_id_drive}...")
       file_metadata = {'name': nombre_archivo_log, 'parents': [folder_id_drive]}
       media = MediaFileUpload(filepath, mimetype='text/csv' if filepath.endswith('.csv') else 'text/plain')
       service.files().create(body=file_metadata, media_body=media, fields='id').execute()
       logger.info(f"Log '{nombre_archivo_log}' subido correctamente.")
   except Exception as e:
       logger.error(f"Error al subir el log '{filepath}' a Drive: {e}")


# ========== FUNCIÓN PRINCIPAL PARA RENOMBRAR ==========
def descargar_y_renombrar(service_drive, usar_ia_renombrado=True): # Cambiados nombres de parámetros
   """Descarga y renombra los PDFs de facturas en Drive."""
   resultados_renombrado = []
   try:
        archivos_en_drive = service_drive.files().list(
            q=f"'{FOLDER_ID}' in parents and (mimeType='application/pdf' or mimeType contains 'image/') and trashed=false", # Añadido trashed=false
            fields="files(id, name, mimeType)",
            pageSize=1000 # Máximo permitido
        ).execute().get('files', [])
   except Exception as e:
       logger.error(f"Error al listar archivos de Google Drive: {e}")
       return []


   if not archivos_en_drive:
       logger.info("No se encontraron archivos PDF o de imagen en la carpeta de Drive.")
       return []

   # Eliminar duplicados si Google Drive API devuelve alguno (basado en id)
   archivos_unicos_drive = {archivo['id']: archivo for archivo in archivos_en_drive}
   archivos_a_procesar = list(archivos_unicos_drive.values())
   logger.info(f"Se encontraron {len(archivos_a_procesar)} archivos PDF/Imagen únicos en la carpeta de Drive.")

   renombrados_count = 0
   errores_count = 0
   lista_datos_archivos = []

   for i, archivo_drive in enumerate(archivos_a_procesar):
       file_id_drive = archivo_drive['id']
       nombre_original_drive = archivo_drive['name']
       mime_type_drive = archivo_drive['mimeType']
       log_prefix = f"[{i+1}/{len(archivos_a_procesar)}] '{nombre_original_drive}'"

       if re.match(PATTERN_RENAMED, nombre_original_drive):
           logger.info(f"{log_prefix}: ✅ Ya está en formato renombrado. Extrayendo datos del nombre...")
           datos_nombre = extraer_datos_desde_nombre(nombre_original_drive)
           if datos_nombre:
               # Descargar para obtener texto completo si es necesario para datos adicionales
               try:
                   request_media = service_drive.files().get_media(fileId=file_id_drive)
                   fh_media = io.BytesIO()
                   downloader = MediaIoBaseDownload(fh_media, request_media)
                   done_download = False
                   while not done_download:
                       _, done_download = downloader.next_chunk()
                   fh_media.seek(0)
                   
                   if mime_type_drive == 'application/pdf':
                       texto1 = leer_pagina(fh_media, 0)
                       texto2 = leer_pagina(fh_media, 1)
                       datos_nombre['texto_completo'] = (texto1 + "\n" + texto2).strip()
                   elif mime_type_drive.startswith('image/'):
                        fh_media.seek(0) # Asegurar que esté al inicio para Image.open
                        img_pil_temp = Image.open(fh_media)
                        datos_nombre['texto_completo'] = pytesseract.image_to_string(img_pil_temp, lang='spa')
                   else: # No debería pasar por el filtro inicial
                       datos_nombre['texto_completo'] = ""
                   
                   datos_nombre['cuil_afiliado'] = extraer_cuil_afiliado(datos_nombre['texto_completo'])
                   datos_nombre['nombre_afiliado'] = extraer_nombre_afiliado(datos_nombre['texto_completo'])
                   datos_nombre['file_id'] = file_id_drive # Importante para después
                   lista_datos_archivos.append(datos_nombre)
                   logger.info(f"{log_prefix}:   ✅ Datos extraídos del nombre y contenido.")
               except Exception as e_download_renamed:
                   logger.error(f"{log_prefix}:   ❌ Error al descargar/procesar contenido de archivo ya renombrado: {e_download_renamed}")
                   # Guardamos al menos los datos del nombre
                   datos_nombre['file_id'] = file_id_drive 
                   lista_datos_archivos.append(datos_nombre)

           else: # No se pudieron extraer datos del nombre (raro si el patrón coincide)
               logger.warning(f"{log_prefix}:   ⚠️ No se pudieron extraer datos del nombre a pesar de coincidir con el patrón.")
           continue # Pasar al siguiente archivo

       logger.info(f"\n{log_prefix}: Procesando para renombrar...")
       try:
           request_media = service_drive.files().get_media(fileId=file_id_drive)
           fh_media = io.BytesIO()
           downloader = MediaIoBaseDownload(fh_media, request_media)
           done_download = False
           while not done_download:
               _, done_download = downloader.next_chunk()
           fh_media.seek(0) # Esencial para relecturas
       except Exception as e_download:
           logger.error(f"{log_prefix}: ❌ Error al descargar: {e_download}")
           errores_count +=1
           resultados_renombrado.append([nombre_original_drive, '', '❌ ERROR DESCARGA', str(e_download)])
           continue

       datos_extraidos_contenido = None
       error_extraccion = None
       
       if mime_type_drive == 'application/pdf':
           datos_extraidos_contenido, error_extraccion = extraer_desde_dos_paginas(fh_media, nombre_original_drive, usar_ia_renombrado)
       elif mime_type_drive.startswith('image/'):
           datos_extraidos_contenido, error_extraccion = extraer_desde_imagen(fh_media.getvalue(), nombre_original_drive, usar_ia_renombrado)
       else:
           resultados_renombrado.append([nombre_original_drive, '', '⚠️ IGNORADO', 'Tipo de archivo no admitido'])
           logger.info(f"{log_prefix}: ⚠️ Tipo de archivo '{mime_type_drive}' no admitido para procesamiento.")
           continue

       if error_extraccion or not datos_extraidos_contenido or not datos_extraidos_contenido.get('nombre_archivo'):
           errores_count += 1
           msg_error = error_extraccion if error_extraccion else "No se pudo generar nombre de archivo"
           resultados_renombrado.append([nombre_original_drive, '', '❌ ERROR EXTRACCIÓN', msg_error])
           logger.error(f"{log_prefix}:   ❌ {msg_error}")
           continue

       nuevo_nombre_generado = datos_extraidos_contenido['nombre_archivo']
       try:
           service_drive.files().update(fileId=file_id_drive, body={'name': nuevo_nombre_generado}).execute()
           renombrados_count += 1
           resultados_renombrado.append([nombre_original_drive, nuevo_nombre_generado, '✅ OK', ''])
           logger.info(f"{log_prefix}:   ✅ Renombrado a: {nuevo_nombre_generado}")
           
           datos_extraidos_contenido['file_id'] = file_id_drive # Guardar ID para TXT
           datos_extraidos_contenido['nombre_original'] = nombre_original_drive # Útil para logs
           lista_datos_archivos.append(datos_extraidos_contenido)
           
       except Exception as e_rename:
           errores_count += 1
           resultados_renombrado.append([nombre_original_drive, nuevo_nombre_generado, '❌ ERROR AL RENOMBRAR EN DRIVE', str(e_rename)])
           logger.error(f"{log_prefix}:   ❌ Error al renombrar en Drive: {e_rename}")
           # Aunque no se pudo renombrar, si los datos se extrajeron, podríamos intentar usarlos para TXT
           # Esto depende de si es crítico que esté renombrado en Drive primero.
           # Por ahora, si no se renombra, no lo añadimos a lista_datos_archivos para TXT
           # para evitar procesar algo que no tiene el nombre esperado en Drive.

   # Generar y subir log de renombrado
   if resultados_renombrado or len(archivos_a_procesar) > 0 : # Solo generar log si se intentó procesar algo
        timestamp_log = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_renombrado_filename = f'log_renombrado_{timestamp_log}.csv'
        try:
            with open(log_renombrado_filename, 'w', newline='', encoding='utf-8') as f_log:
                writer_log = csv.writer(f_log)
                writer_log.writerow(['Nombre original', 'Nombre nuevo', 'Estado', 'Observación'])
                writer_log.writerows(resultados_renombrado)
            logger.info(f"\nLog de renombrado guardado localmente como '{log_renombrado_filename}'")
            subir_log(service_drive, log_renombrado_filename, FOLDER_ID)
        except Exception as e_log:
            logger.error(f"Error al generar o subir el log de renombrado: {e_log}")


   logger.info(f"\n📄 Total de archivos revisados en Drive: {len(archivos_a_procesar)}")
   logger.info(f"✅ Renombrados correctamente en Drive: {renombrados_count}")
   logger.info(f"❌ Con errores (descarga, extracción o renombrado): {errores_count}")
   
   return lista_datos_archivos


# ========== FUNCIONES PARA GENERACIÓN DE TXT ========== (Sección revisada para claridad)
def obtener_excel_drive(service_drive): # Cambiado nombre de parámetro
   """Busca y descarga el archivo Excel desde Drive."""
   logger.info("Buscando archivo Excel en la carpeta de Drive...")
   
   # Buscar archivo con extensión .xlsx o .xls, no trashed
   query_excel = f"'{FOLDER_ID}' in parents and (name contains '.xlsx' or name contains '.xls') and mimeType contains 'spreadsheet' and trashed = false"
   try:
        excel_files_drive = service_drive.files().list(
            q=query_excel,
            fields="files(id, name)",
            orderBy="modifiedTime desc", # Tomar el más reciente si hay varios
            pageSize=10 
        ).execute().get('files', [])
   except Exception as e:
       logger.error(f"Error al buscar archivos Excel en Drive: {e}")
       return None, None
   
   if not excel_files_drive:
       logger.error("No se encontraron archivos Excel en la carpeta de Drive especificada.")
       return None, None
   
   excel_file_seleccionado = excel_files_drive[0] # Tomar el primero (más reciente por orderBy)
   if len(excel_files_drive) > 1:
       logger.warning(f"Se encontraron {len(excel_files_drive)} archivos Excel. Se usará el más reciente: '{excel_file_seleccionado['name']}'")
       # Se podría añadir lógica para que el usuario elija si hay varios.
   
   logger.info(f"Descargando Excel: {excel_file_seleccionado['name']} (ID: {excel_file_seleccionado['id']})")
   try:
       request_excel = service_drive.files().get_media(fileId=excel_file_seleccionado['id'])
       fh_excel = io.BytesIO()
       downloader_excel = MediaIoBaseDownload(fh_excel, request_excel)
       done_excel = False
       while not done_excel:
           status_excel, done_excel = downloader_excel.next_chunk()
           logger.info(f"Descarga Excel {int(status_excel.progress() * 100)}%")
       fh_excel.seek(0)
       return fh_excel, excel_file_seleccionado['name']
   except Exception as e:
       logger.error(f"Error al descargar el archivo Excel '{excel_file_seleccionado['name']}': {e}")
       return None, None


def leer_excel_dataframe(excel_bytes_io): # excel_bytes_io es un io.BytesIO
    """Lee el archivo Excel y devuelve un DataFrame."""
    if not excel_bytes_io: return None
    try:
        # Leer como strings para evitar conversiones automáticas problemáticas
        df_excel = pd.read_excel(excel_bytes_io, dtype=str)
        df_excel.columns = df_excel.columns.str.strip()  # Quitar espacios al inicio/final
        df_excel.columns = df_excel.columns.str.lower()  # Convertir a minúsculas
        df_excel.columns = df_excel.columns.str.replace(' ', '_', regex=False) # Reemplazar espacios con guiones bajos
        
        columnas_requeridas_excel = {"cuil", "codigo_certificado", "vencimiento_certificado", "provincia"}
        if not columnas_requeridas_excel.issubset(df_excel.columns):
            columnas_faltantes_excel = columnas_requeridas_excel - set(df_excel.columns)
            logger.error(f"Faltan columnas requeridas en el Excel: {columnas_faltantes_excel}")
            return None
            
        # Limpieza básica
        for col_excel in df_excel.columns:
            if col_excel in ["cuil", "codigo_certificado", "provincia", "nombre", "nombre_afiliado"]: # Columnas donde quitar espacios extra
                 df_excel[col_excel] = df_excel[col_excel].astype(str).str.strip()
        
        # Formatear fecha de vencimiento
        if 'vencimiento_certificado' in df_excel.columns:
            df_excel['vencimiento_certificado'] = df_excel['vencimiento_certificado'].apply(formatear_fecha)
        
        # Renombrar columna de nombre si es necesario
        if 'nombre_afiliado' in df_excel.columns and 'nombre' not in df_excel.columns:
            df_excel.rename(columns={'nombre_afiliado': 'nombre'}, inplace=True)
        elif 'nombre afiliado' in df_excel.columns and 'nombre' not in df_excel.columns: # Con espacio
            df_excel.rename(columns={'nombre afiliado': 'nombre'}, inplace=True)


        logger.info(f"Excel cargado y procesado con {len(df_excel)} registros.")
        return df_excel
    except Exception as e:
        logger.error(f"Error crítico al procesar el archivo Excel: {e}", exc_info=True)
        return None

def buscar_cuil_afiliado_en_excel(nombre_afiliado_factura, df_excel_afiliados):
    """Busca el CUIL de un afiliado en el DataFrame del Excel usando su nombre."""
    if not nombre_afiliado_factura or df_excel_afiliados is None or 'nombre' not in df_excel_afiliados.columns:
        return None

    import unicodedata
    def normalizar_nombre(texto):
        if not isinstance(texto, str): return ""
        return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII').lower().strip()

    nombre_factura_norm = normalizar_nombre(nombre_afiliado_factura)
    
    # Crear columna normalizada en df_excel si no existe para la comparación
    if 'nombre_norm_excel' not in df_excel_afiliados.columns:
        df_excel_afiliados['nombre_norm_excel'] = df_excel_afiliados['nombre'].apply(normalizar_nombre)

    # Búsqueda exacta primero
    coincidencias_exactas = df_excel_afiliados[df_excel_afiliados['nombre_norm_excel'] == nombre_factura_norm]
    if not coincidencias_exactas.empty:
        return coincidencias_exactas.iloc[0]['cuil'] # Devuelve el CUIL de la primera coincidencia exacta

    # Búsqueda parcial si no hay exacta (ej. apellido y primera letra del nombre)
    # Esta lógica puede ser compleja y depende de la calidad de los datos.
    # Por ahora, se omite la búsqueda parcial compleja para simplificar.
    # Si se necesita, se puede implementar aquí.
    # Ejemplo: si nombre_factura_norm es "Perez Juan", buscar "Perez J" o "Juan Perez"

    logger.debug(f"No se encontró coincidencia exacta en Excel para el nombre normalizado: '{nombre_factura_norm}'")
    return None


def generar_txt_final(service_drive, lista_datos_facturas, rnos_txt="000000", usar_ia_txt=True): # Parámetros renombrados
    """Genera el archivo TXT final."""
    logger.info("\n========== INICIANDO GENERACIÓN DE ARCHIVO TXT FINAL ==========")
    
    rnos_txt_validado = ''.join(c for c in str(rnos_txt) if c.isdigit()).zfill(6)[:6]
    logger.info(f"RNOS para TXT: {rnos_txt_validado}")
    
    excel_bytes_io, _ = obtener_excel_drive(service_drive)
    if not excel_bytes_io: return
    
    df_afiliados_excel = leer_excel_dataframe(excel_bytes_io)
    if df_afiliados_excel is None: return

    lineas_para_txt = []
    procesados_ok_txt = 0
    errores_txt = 0

    logger.info(f"Procesando {len(lista_datos_facturas)} facturas para el TXT...")
    for i, datos_factura_base in enumerate(lista_datos_facturas):
        log_prefix_txt = f"[Factura {i+1}/{len(lista_datos_facturas)}: {datos_factura_base.get('nombre_archivo', 'Desconocida')}]"
        logger.info(f"{log_prefix_txt} Iniciando procesamiento para TXT...")

        info_completa_factura_para_txt = extraer_datos_adicionales(datos_factura_base, usar_ia_txt)
        if not info_completa_factura_para_txt:
            logger.error(f"{log_prefix_txt} ❌ No se pudieron obtener datos completos para el TXT.")
            errores_txt += 1
            continue

        fila_afiliado_excel = None
        cuil_afiliado_encontrado_en_excel = None

        # Intento 1: Usar CUIL del afiliado si la IA lo extrajo de la factura
        cuil_extraido_factura = info_completa_factura_para_txt.get('cuil_afiliado')
        if cuil_extraido_factura and len(str(cuil_extraido_factura)) == 11:
            # Asegurar que df_afiliados_excel['cuil'] sea string para la comparación
            df_afiliados_excel['cuil_str'] = df_afiliados_excel['cuil'].astype(str).str.strip()
            coincidencia_excel = df_afiliados_excel[df_afiliados_excel['cuil_str'] == str(cuil_extraido_factura).strip()]
            if not coincidencia_excel.empty:
                fila_afiliado_excel = coincidencia_excel.iloc[0]
                cuil_afiliado_encontrado_en_excel = fila_afiliado_excel['cuil'] # Guardar el CUIL original del Excel
                logger.info(f"{log_prefix_txt}  ✅ Afiliado encontrado en Excel por CUIL extraído de factura: {cuil_afiliado_encontrado_en_excel}")

        # Intento 2: Usar DNI del afiliado si la IA lo extrajo y no se encontró por CUIL
        if fila_afiliado_excel is None:
            dni_extraido_factura = info_completa_factura_para_txt.get('dni_afiliado') # Asumiendo que 'dni_afiliado' es el campo
            if dni_extraido_factura and len(str(dni_extraido_factura)) >= 7 and len(str(dni_extraido_factura)) <= 8 : # DNI suele tener 7 u 8 dígitos
                dni_factura_str = str(dni_extraido_factura).strip().zfill(8) # Normalizar a 8 dígitos con ceros a la izquierda si es necesario
                
                # Extraer DNI de la columna CUIL en el DataFrame del Excel
                # Un CUIL es XX-DNI-Y. El DNI está entre los guiones o del 3er al 10mo dígito si no hay guiones.
                def get_dni_from_cuil(cuil_str):
                    cuil_limpio = str(cuil_str).replace("-","").replace(" ","")
                    if len(cuil_limpio) == 11:
                        return cuil_limpio[2:10] # Extrae los 8 dígitos del DNI
                    return None

                df_afiliados_excel['dni_from_cuil'] = df_afiliados_excel['cuil'].apply(get_dni_from_cuil)
                coincidencia_excel_dni = df_afiliados_excel[df_afiliados_excel['dni_from_cuil'] == dni_factura_str]
                
                if not coincidencia_excel_dni.empty:
                    if len(coincidencia_excel_dni) == 1: # Solo si hay una única coincidencia por DNI
                        fila_afiliado_excel = coincidencia_excel_dni.iloc[0]
                        cuil_afiliado_encontrado_en_excel = fila_afiliado_excel['cuil']
                        logger.info(f"{log_prefix_txt}  ✅ Afiliado encontrado en Excel por DNI ('{dni_factura_str}' de factura -> CUIL Excel: {cuil_afiliado_encontrado_en_excel})")
                    else:
                        logger.warning(f"{log_prefix_txt}  ⚠️ Múltiples afiliados en Excel con DNI '{dni_factura_str}'. No se puede asignar automáticamente.")


        # Intento 3: Usar nombre del afiliado si la IA lo extrajo y no se encontró por CUIL o DNI
        if fila_afiliado_excel is None: 
            nombre_extraido_factura = info_completa_factura_para_txt.get('nombre_afiliado')
            if nombre_extraido_factura:
                cuil_por_nombre = buscar_cuil_afiliado_en_excel(nombre_extraido_factura, df_afiliados_excel)
                if cuil_por_nombre:
                    # Volver a buscar la fila completa usando el CUIL encontrado por nombre
                    df_afiliados_excel['cuil_str'] = df_afiliados_excel['cuil'].astype(str).str.strip()
                    coincidencia_excel_nombre = df_afiliados_excel[df_afiliados_excel['cuil_str'] == str(cuil_por_nombre).strip()]
                    if not coincidencia_excel_nombre.empty:
                        fila_afiliado_excel = coincidencia_excel_nombre.iloc[0]
                        cuil_afiliado_encontrado_en_excel = fila_afiliado_excel['cuil']
                        logger.info(f"{log_prefix_txt}  ✅ Afiliado encontrado en Excel por NOMBRE extraído de factura ('{nombre_extraido_factura}' -> CUIL Excel: {cuil_afiliado_encontrado_en_excel})")
        
        if fila_afiliado_excel is None:
            logger.warning(f"{log_prefix_txt}  ⚠️ No se pudo encontrar un afiliado en el Excel para esta factura. Se omitirá este registro del TXT.")
            errores_txt += 1
            continue
            
        # Paso 3: Construir la línea del TXT
        # Asegúrate de pasar el CUIL correcto (el del Excel) a construir_linea si es necesario,
        # o que construir_linea use el CUIL de fila_afiliado_excel.
        # La función construir_linea ya espera fila_excel_afiliado.
        linea_txt_generada = construir_linea(rnos_txt_validado, fila_afiliado_excel, info_completa_factura_para_txt)
        
        if linea_txt_generada:
            lineas_para_txt.append(linea_txt_generada)
            procesados_ok_txt += 1
            logger.info(f"{log_prefix_txt}   -> Línea TXT generada para CUIL {cuil_afiliado_encontrado_en_excel}.") # Usar el CUIL encontrado
        else: 
            logger.error(f"{log_prefix_txt} ❌ Error al construir la línea TXT para CUIL {cuil_afiliado_encontrado_en_excel}.")
            errores_txt += 1

    if not lineas_para_txt:
        logger.error("No se generaron líneas válidas para el archivo TXT. Proceso abortado.")
        return

    nombre_archivo_txt = f"{rnos_txt_validado}_ds.txt"
    try:
        with open(nombre_archivo_txt, "w", encoding="utf-8", newline="\r\n") as f_txt: # Usar CRLF
            f_txt.write("\n".join(lineas_para_txt))
        logger.info(f"Archivo TXT generado localmente: {nombre_archivo_txt} ({len(lineas_para_txt)} líneas)")
    except Exception as e_write_txt:
        logger.error(f"Error al escribir el archivo TXT local '{nombre_archivo_txt}': {e_write_txt}")
        return

    if verificar_formato_txt(nombre_archivo_txt):
        logger.info(f"Subiendo TXT '{nombre_archivo_txt}' a Google Drive...")
        try:
            file_metadata_txt = {'name': nombre_archivo_txt, 'parents': [FOLDER_ID], 'mimeType': 'text/plain'}
            media_txt = MediaFileUpload(nombre_archivo_txt, mimetype='text/plain', resumable=True)
            service_drive.files().create(body=file_metadata_txt, media_body=media_txt, fields='id').execute()
            logger.info(f"Archivo TXT '{nombre_archivo_txt}' subido a Drive correctamente.")
        except Exception as e_upload_txt:
            logger.error(f"Error al subir el archivo TXT '{nombre_archivo_txt}' a Drive: {e_upload_txt}")
    else:
        logger.warning(f"El archivo TXT '{nombre_archivo_txt}' tiene errores de formato y NO se subirá a Drive. Por favor, revíselo manualmente.")


    logger.info("\n" + "="*50)
    logger.info(f"RESUMEN FINAL GENERACIÓN TXT:")
    logger.info(f"- RNOS: {rnos_txt_validado}")
    logger.info(f"- Total de facturas de entrada: {len(lista_datos_facturas)}")
    logger.info(f"- Registros procesados y escritos en TXT: {procesados_ok_txt}")
    logger.info(f"- Registros con error o sin afiliado coincidente: {errores_txt}")
    logger.info("="*50)


# ========== EJECUCIÓN PRINCIPAL ==========
def main():
    """Función principal que ejecuta todo el proceso."""
    # Mover la declaración global al principio de la función
    global DEVICE, MODEL_PATH, factura_processor # factura_processor también es global
    
    # === INICIO DE CÓDIGO DE DEPURACIÓN AÑADIDO ===
    # Para depurar errores de CUDA, esto puede dar un stacktrace más útil
    # Debe ir al principio de la ejecución del script o de la función main.
    if os.environ.get("DEBUG_CUDA") == "1" or ("cuda" in DEVICE.type): # Solo si se quiere depurar CUDA o si se va a usar GPU
        logger.info("Intentando establecer CUDA_LAUNCH_BLOCKING=1 para depuración de GPU.")
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # === FIN DE CÓDIGO DE DEPURACIÓN AÑADIDO ===
    
    try:
        print("\n" + "="*50)
        print("   SISTEMA UNIFICADO DE FACTURAS - OBRA SOCIAL")
        print("="*50)
        print("Este programa realiza dos tareas:")
        print("1. Renombrar archivos PDF/Imagen según patrón CUIT_CODIGO_PV_NRO")
        print("2. Generar archivo TXT para presentación en obra social")
        print("-"*50)
        
        usar_ia_input = input("¿Desea utilizar IA para mejorar la extracción de datos? (s/n) [s]: ").lower() or "s"
        usar_ia = usar_ia_input == 's'
        
        if usar_ia:
            if torch.cuda.is_available():
                forzar_cpu_input = input("GPU disponible. ¿Desea forzar el uso de CPU (útil si hay problemas con GPU)? (s/n) [n]: ").lower() or "n"
                if forzar_cpu_input == 's':
                    DEVICE = torch.device("cpu")
                    logger.info("Forzando uso de CPU para el modelo de IA.")
                else:
                    DEVICE = torch.device("cuda") # Asegurar que sea cuda si no se fuerza CPU
            else: # GPU no disponible
                DEVICE = torch.device("cpu")
                logger.info("GPU no disponible. Usando CPU para el modelo de IA.")

            # La ruta al modelo fine-tuneado es relativa a la carpeta `fine_tuning_facturas`
            # que debe estar al mismo nivel que este script, o el script debe estar dentro de ella.
            # Si `prueba_bot_final.py` está en la raíz del proyecto, y `fine_tuning_facturas` es una subcarpeta:
            path_base_script = os.path.dirname(os.path.abspath(__file__)) # Directorio donde está prueba_bot_final.py
            path_modelo_sugerido = os.path.join(path_base_script, "fine_tuning_facturas", "layoutlmv3-finetuned-facturas_final")
            
            logger.info(f"Buscando modelo fine-tuneado en: {path_modelo_sugerido}")

            if os.path.exists(path_modelo_sugerido):
                usar_fine_tuned_input = input(f"Modelo fine-tuneado encontrado. ¿Desea usarlo? (s/n) [s]: ").lower() or "s"
                if usar_fine_tuned_input == 's':
                    MODEL_PATH = path_modelo_sugerido
                else:
                    MODEL_PATH = "microsoft/layoutlmv3-base"
            else:
                logger.warning(f"¡No se encontró el modelo fine-tuneado en {path_modelo_sugerido}!")
                MODEL_PATH = "microsoft/layoutlmv3-base"
            
            logger.info(f"Se usará MODEL_PATH: {MODEL_PATH} en DEVICE: {DEVICE}")
            inicializar_modelo() # factura_processor se inicializa aquí
            if factura_processor:
                 logger.info("Sistema de IA inicializado correctamente.")
            else:
                 logger.error("¡FALLO AL INICIALIZAR EL SISTEMA DE IA! Verifique logs.")
                 usar_ia = False # Desactivar IA si no se pudo inicializar
        else:
            logger.info("Extracción de datos solo con métodos tradicionales (sin IA).")
            factura_processor = None 
        
        logger.info("Autenticando con Google Drive...")
        service = autenticar() # `service` ahora es el servicio de Drive
        logger.info("Autenticación con Google Drive exitosa.")
        
        logger.info("\n=== INICIANDO ETAPA 1: DESCARGAR Y RENOMBRAR ARCHIVOS ===\n")
        archivos_datos_procesados = descargar_y_renombrar(service, usar_ia) # Pasar el servicio de drive
        
        if not archivos_datos_procesados:
            logger.warning("No se procesaron archivos en la etapa de renombrado o no se encontraron archivos válidos.")
            logger.info("Buscando si existen archivos ya renombrados en Drive para intentar generar el TXT...")
            
            archivos_drive_existentes = []
            try:
                archivos_drive_existentes_list = service.files().list(
                    q=f"'{FOLDER_ID}' in parents and mimeType='application/pdf' and trashed = false",
                    fields="files(id, name)",
                    pageSize=1000 
                ).execute().get('files', [])
                if archivos_drive_existentes_list:
                    archivos_drive_existentes = archivos_drive_existentes_list
            except Exception as e_list_drive:
                logger.error(f"Error al listar archivos de Drive para fallback: {e_list_drive}")

            archivos_renombrados_para_txt = []
            if archivos_drive_existentes:
                for archivo_info_drive in archivos_drive_existentes:
                    nombre_archivo_en_drive = archivo_info_drive['name']
                    if re.match(PATTERN_RENAMED, nombre_archivo_en_drive):
                        datos_desde_nombre_drive = extraer_datos_desde_nombre(nombre_archivo_en_drive)
                        if datos_desde_nombre_drive:
                            # Para estos, necesitaríamos el texto_completo si la IA no lo procesó
                            # Esto es un fallback, así que puede que no tengamos texto completo
                            # Si no hay texto_completo, la extracción de datos adicionales para TXT será limitada
                            datos_desde_nombre_drive['file_id'] = archivo_info_drive['id']
                            datos_desde_nombre_drive['texto_completo'] = "" # Placeholder
                            # Podríamos intentar descargar y leer el texto aquí si es crucial
                            logger.info(f"Archivo ya renombrado encontrado para TXT: {nombre_archivo_en_drive}. Texto completo no extraído en este fallback.")
                            archivos_renombrados_para_txt.append(datos_desde_nombre_drive)
            
            if archivos_renombrados_para_txt:
                logger.info(f"Se usarán {len(archivos_renombrados_para_txt)} archivos ya renombrados de Drive para la generación de TXT.")
                archivos_datos_procesados = archivos_renombrados_para_txt
            else:
                logger.error("No hay archivos válidos (ni procesados ni ya renombrados) para la generación de TXT.")
                input("\nPresione Enter para salir...")
                return 
        
        logger.info("\n=== INICIANDO ETAPA 2: GENERAR TXT PARA OBRA SOCIAL ===\n")
        rnos_usuario = ""
        while not (rnos_usuario.isdigit() and len(rnos_usuario) == 6) :
            rnos_usuario = input("Ingrese el RNOS de la obra social (exactamente 6 dígitos): ").strip()
            if not (rnos_usuario.isdigit() and len(rnos_usuario) == 6):
                logger.error("RNOS inválido. Debe contener 6 dígitos.")

        generar_txt_final(service, archivos_datos_procesados, rnos_usuario, usar_ia) 
        
        logger.info("\n¡Proceso completado!")
    
    except FileNotFoundError as e_fnf:
        logger.error(f"Error de archivo no encontrado: {e_fnf}. Asegúrese de que 'credentials.json' esté presente.")
    except Exception as e_main: 
        logger.error(f"Error general en la ejecución del script: {e_main}", exc_info=True) # exc_info para traceback completo
       
# ========== EJECUCIÓN DEL SCRIPT ==========
if __name__ == '__main__':
   try:
      main()
   except KeyboardInterrupt:
       logger.info("\nProceso interrumpido por el usuario (Ctrl+C).")
   except Exception as e_global:
       logger.critical(f"Error CRÍTICO e inesperado fuera de la función main: {e_global}", exc_info=True)
   finally:
       logger.info("Fin del script.")
       input("\nPresione Enter para salir...")