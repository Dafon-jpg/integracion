import json
import os
import torch
import numpy as np
import inspect
from PIL import Image
import transformers
from datasets import load_dataset, Features, Sequence, Value, Array2D, ClassLabel
from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3TokenizerFast,
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification
)
# Importar TrainingArguments desde el módulo específico
from transformers.training_args import TrainingArguments
from seqeval.metrics import classification_report

# Verificar versión de transformers
print(f"Versión de transformers: {transformers.__version__}")

# Verificar parámetros aceptados por TrainingArguments
print("Parámetros aceptados por TrainingArguments:")
signature = inspect.signature(TrainingArguments.__init__)
for param_name, param in signature.parameters.items():
    if param_name != 'self':
        print(f"- {param_name}")

# --- CONFIGURACIÓN DEL ENTRENAMIENTO ---

# 1. Rutas a tus archivos de dataset (generados por dividir_dataset.py)
archivo_train_dataset = "train.jsonl"
archivo_eval_dataset = "eval.jsonl"

# 2. Nombre del modelo base de Hugging Face
nombre_modelo_base = "microsoft/layoutlmv3-base"

# 3. Directorio donde se guardará el modelo fine-tuneado y los checkpoints
directorio_salida_modelo = "./layoutlmv3-finetuned-facturas"

# 4. Tus etiquetas BASE (sin prefijos B-/I-) - ¡DEBE COINCIDIR CON PREPARAR_DATASET.PY!
BASE_LABELS = sorted([
    "CUIT_PRESTADOR", "CUIT_AFILIADO", "NOMBRE_AFILIADO", "NOMBRE_PRESTADOR",
    "TIPO_FACTURA", "LETRA_FACTURA", "PUNTO_VENTA", "NUMERO_FACTURA",
    "FECHA_EMISION", "CAE", "IMPORTE", "PERIODO", "ACTIVIDAD", "DNI_AFILIADO"
])

# 5. Hiperparámetros del entrenamiento
NUM_TRAIN_EPOCHS = 15       
PER_DEVICE_TRAIN_BATCH_SIZE = 2 
PER_DEVICE_EVAL_BATCH_SIZE = 2
LEARNING_RATE = 4e-5      
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
WARMUP_RATIO = 0.1        
GRADIENT_ACCUMULATION_STEPS = 4

# 6. Habilitar entrenamiento con precisión mixta si hay GPU
FP16_TRAINING = torch.cuda.is_available()

# 7. Tu token de Hugging Face
AUTH_TOKEN_HF = "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" 

if not AUTH_TOKEN_HF.startswith("hf_"):
    print("\n!!! ADVERTENCIA: Token de Hugging Face no configurado correctamente !!!\n")
    print("El token debe empezar con 'hf_'. Ve a https://huggingface.co/settings/tokens para obtener uno.")
    exit("Error: Token de Hugging Face incorrecto en el script.")
# --- FIN CONFIGURACIÓN ---


# --- Preparar Lista de Etiquetas y Mapeos ---
label_list_iob = ["O"]
for label_name in BASE_LABELS:
    label_list_iob.append(f"B-{label_name}")
    label_list_iob.append(f"I-{label_name}")

label2id = {label: i for i, label in enumerate(label_list_iob)}
id2label = {i: label for i, label in enumerate(label_list_iob)}
num_labels_total = len(label_list_iob)

print(f"Lista de etiquetas IOB (total {num_labels_total}): {label_list_iob}")
print(f"Mapeo label2id (primeros 5): {list(label2id.items())[:5]}")
# --- Fin Preparar Etiquetas ---

# --- Cargar Datasets ---
features = Features({
    'id': Value('string'),
    'words': Sequence(Value('string')),
    'bboxes': Sequence(Sequence(Value('int64'))), 
    'ner_tags': Sequence(ClassLabel(names=label_list_iob)),
    'image_path': Value('string')
})

data_files = {
    "train": archivo_train_dataset,
    "validation": archivo_eval_dataset
}
print(f"Cargando datasets desde: {data_files}")
try:
    if not os.path.exists(data_files["train"]):
        raise FileNotFoundError(f"Archivo de entrenamiento no encontrado: {data_files['train']}")
    if not os.path.exists(data_files["validation"]):
        raise FileNotFoundError(f"Archivo de validación no encontrado: {data_files['validation']}")
        
    raw_datasets = load_dataset("json", data_files=data_files, features=features)
    print(f"Datasets cargados: {raw_datasets}")
except Exception as e:
    print(f"Error al cargar datasets: {e}")
    print("Asegúrate de que los archivos train.jsonl y eval.jsonl existan, estén en esta carpeta, y tengan el formato correcto.")
    exit()
# --- Fin Cargar Datasets ---

# --- Inicializar Procesador y Modelo ---
print(f"Intentando cargar el procesador y modelo '{nombre_modelo_base}' usando token...")
try:
    processor = LayoutLMv3Processor.from_pretrained(
        nombre_modelo_base, 
        apply_ocr=False,
        token=AUTH_TOKEN_HF
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        nombre_modelo_base,
        num_labels=num_labels_total,
        id2label=id2label,
        label2id=label2id,
        token=AUTH_TOKEN_HF
    )
    print(f"Procesador y modelo '{nombre_modelo_base}' cargados correctamente.")
except Exception as e:
    print(f"Error al cargar el procesador o el modelo: {e}")
    print("Verifica tu token, conexión a internet y el nombre del modelo base.")
    exit()
# --- Fin Inicializar ---


# --- Función de Preprocesamiento para el Dataset ---
def preprocess_data_for_model(examples, max_len=512):
    try:
        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    except Exception as e:
        print(f"Error cargando imágenes durante el preprocesamiento: {e}")
        raise e

    tokenized_inputs = processor(
        images=images,
        text=examples['words'], 
        boxes=examples['bboxes'], 
        word_labels=examples['ner_tags'], 
        padding="max_length",
        truncation=True,
        max_length=max_len, 
    )
    return tokenized_inputs

print("Preprocesando datasets...")
try:
    processed_datasets = raw_datasets.map(
        preprocess_data_for_model,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={"max_len": 512} 
    )
    print(f"Datasets preprocesados. Ejemplo de train (features): {processed_datasets['train'].features}")
except Exception as e:
    print(f"Error durante el preprocesamiento de datos: {e}")
    print("Revisa la función de preprocesamiento y los archivos .jsonl.")
    exit()
# --- Fin Preprocesamiento ---

# --- Data Collator ---
data_collator = DataCollatorForTokenClassification(tokenizer=processor.tokenizer)
# --- Fin Data Collator ---

# --- Definir Argumentos de Entrenamiento (VERSIÓN ULTRA BÁSICA) ---
print("Definiendo argumentos de entrenamiento...")
try:
    # Usar solo argumentos básicos
    training_args = TrainingArguments(
        output_dir=directorio_salida_modelo,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        # No usar argumentos que puedan causar problemas
    )
    print("Argumentos de entrenamiento definidos correctamente")
except Exception as e:
    print(f"Error al definir argumentos de entrenamiento: {e}")
    import traceback
    traceback.print_exc()
    exit()
# --- Fin Argumentos de Entrenamiento ---

# --- Función para Calcular Métricas ---
def compute_metrics_for_layoutlm(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2) 
    true_predictions = [
        [id2label[p_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
        for prediction, label in zip(predictions, labels)
    ]
    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
    return {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }
# --- Fin Calcular Métricas ---

# --- Inicializar Trainer ---
print("Inicializando Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    tokenizer=processor.tokenizer, 
    data_collator=data_collator,
    compute_metrics=compute_metrics_for_layoutlm,
)
# --- Fin Inicializar Trainer ---

# --- ¡Entrenar el Modelo! ---
print("\n--- ¡INICIANDO FINE-TUNING! ---")
try:
    trainer.train()
    print("--- ¡FINE-TUNING COMPLETADO! ---")

    print(f"Guardando modelo final en: {directorio_salida_modelo}_final")
    os.makedirs(f"{directorio_salida_modelo}_final", exist_ok=True)
    trainer.save_model(f"{directorio_salida_modelo}_final")
    processor.save_pretrained(f"{directorio_salida_modelo}_final")

    print("\nEvaluando el modelo final en el conjunto de evaluación...")
    eval_results = trainer.evaluate()
    print(f"Resultados de la evaluación final: {eval_results}")

except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
    import traceback
    traceback.print_exc()

print("\nProceso de fine-tuning finalizado.")
print(f"Puedes encontrar tu modelo fine-tuneado en la carpeta: '{directorio_salida_modelo}_final'")
print(f"Y los checkpoints del entrenamiento en: '{directorio_salida_modelo}'")