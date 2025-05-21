import json
import random
import os # Añadido para construir la ruta al directorio del script

# Obtener la ruta del directorio donde está este script
DIRECTORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))

archivo_entrada = os.path.join(DIRECTORIO_SCRIPT, "dataset_entrenamiento.jsonl") # Asegura que lee desde la misma carpeta
archivo_train = os.path.join(DIRECTORIO_SCRIPT, "train.jsonl")
archivo_eval = os.path.join(DIRECTORIO_SCRIPT, "eval.jsonl")

porcentaje_eval = 0.15 # Usar 15% para evaluación (aprox 8 de 52)

lineas = []
if not os.path.exists(archivo_entrada):
    print(f"ERROR: El archivo de entrada '{archivo_entrada}' no existe. Ejecuta preparar_dataset.py primero.")
    exit()

with open(archivo_entrada, 'r', encoding='utf-8') as f:
    for line in f:
        lineas.append(json.loads(line))

if not lineas:
    print(f"ERROR: El archivo de entrada '{archivo_entrada}' está vacío.")
    exit()

random.seed(42) # Añadir una semilla para que la división sea reproducible
random.shuffle(lineas) # Mezclar aleatoriamente

punto_division = int(len(lineas) * (1 - porcentaje_eval))

train_data = lineas[:punto_division]
eval_data = lineas[punto_division:]

with open(archivo_train, 'w', encoding='utf-8') as f:
    for entrada in train_data:
        f.write(json.dumps(entrada, ensure_ascii=False) + '\n')

with open(archivo_eval, 'w', encoding='utf-8') as f:
    for entrada in eval_data:
        f.write(json.dumps(entrada, ensure_ascii=False) + '\n')

print(f"Dataset dividido:")
print(f"  Total de ejemplos originales: {len(lineas)}")
print(f"  Entrenamiento ({len(train_data)} ejemplos) guardado en: {archivo_train}")
print(f"  Evaluación ({len(eval_data)} ejemplos) guardado en: {archivo_eval}")