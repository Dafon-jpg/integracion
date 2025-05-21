import json
import os
from PIL import Image
import urllib.parse
import pytesseract # Asegúrate de tener esto si copias la función

# --- INICIO: Función extraer_palabras_y_cajas (copiada de tu script anterior) ---
# Si tienes Tesseract instalado y en el PATH, no necesitas tesseract_cmd_path
# Si no, debes configurar la variable tesseract_cmd_path más abajo en la sección de configuración.
def extraer_palabras_y_cajas(ruta_imagen, tesseract_cmd_path=None, lang='spa'):
    try:
        if tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
        
        img = Image.open(ruta_imagen)
        ocr_result = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
        
        palabras = []
        coordenadas = [] 
        n_boxes = len(ocr_result['text'])
        for i in range(n_boxes):
            if ocr_result['level'][i] == 5: 
                texto_palabra = ocr_result['text'][i].strip()
                try:
                    confianza_str = str(ocr_result['conf'][i])
                    if '.' in confianza_str: 
                        confianza = int(float(confianza_str))
                    else: 
                        confianza = int(confianza_str)
                except ValueError:
                    confianza = -1 
                if texto_palabra and confianza > 30: 
                    x = ocr_result['left'][i]
                    y = ocr_result['top'][i]
                    w = ocr_result['width'][i]
                    h = ocr_result['height'][i]
                    palabras.append(texto_palabra)
                    coordenadas.append([x, y, x + w, y + h])
        
        if not palabras:
            print(f"    ADVERTENCIA: No se pudieron extraer palabras de {os.path.basename(ruta_imagen)} con la confianza suficiente.")
        return palabras, coordenadas
    except FileNotFoundError:
        print(f"    ERROR: No se encontró la imagen en la ruta: {ruta_imagen}")
        return None, None
    except Exception as e:
        print(f"    ERROR en extraer_palabras_y_cajas para {os.path.basename(ruta_imagen)}: {e}")
        return None, None
# --- FIN: Función extraer_palabras_y_cajas ---


# --- CONFIGURACIÓN PRINCIPAL ---
ruta_json_label_studio = "anotacion_label.json" 
carpeta_base_imagenes_fisicas = r"C:\Users\Usuario\Renombrador de archivos\imagenes_facturas"
archivo_salida_dataset = "dataset_entrenamiento.jsonl"
BASE_LABELS = sorted([ 
    "CUIT_PRESTADOR", "CUIT_AFILIADO", "NOMBRE_AFILIADO", "NOMBRE_PRESTADOR",
    "TIPO_FACTURA", "LETRA_FACTURA", "PUNTO_VENTA", "NUMERO_FACTURA",
    "FECHA_EMISION", "CAE", "IMPORTE", "PERIODO", "ACTIVIDAD", "DNI_AFILIADO"
])
ruta_tesseract_exe = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# --- FIN CONFIGURACIÓN ---

# --- Generar mapeos de etiquetas ---
label_list = ["O"] 
for label_name in BASE_LABELS: 
    label_list.append(f"B-{label_name}")
    label_list.append(f"I-{label_name}")
label2id = {label_name: i for i, label_name in enumerate(label_list)}
id2label = {i: label_name for i, label_name in enumerate(label_list)}
# --- FIN CONFIGURACIÓN ---

def convertir_coordenadas_ls_a_pixeles(ls_value, original_width, original_height):
    x_norm = ls_value['x'] / 100.0
    y_norm = ls_value['y'] / 100.0
    w_norm = ls_value['width'] / 100.0
    h_norm = ls_value['height'] / 100.0
    x1 = int(x_norm * original_width)
    y1 = int(y_norm * original_height)
    x2 = int((x_norm + w_norm) * original_width)
    y2 = int((y_norm + h_norm) * original_height)
    return [x1, y1, x2, y2]

def normalizar_coordenadas_ocr_a_1000(ocr_box, img_width, img_height):
    x1, y1, x2, y2 = ocr_box
    norm_x1 = min(max(0, int((x1 / img_width) * 1000)), 1000)
    norm_y1 = min(max(0, int((y1 / img_height) * 1000)), 1000)
    norm_x2 = min(max(0, int((x2 / img_width) * 1000)), 1000)
    norm_y2 = min(max(0, int((y2 / img_height) * 1000)), 1000)
    if norm_x1 > norm_x2: norm_x1, norm_x2 = norm_x2, norm_x1
    if norm_y1 > norm_y2: norm_y1, norm_y2 = norm_y2, norm_y1
    return [norm_x1, norm_y1, norm_x2, norm_y2]

def calcular_iou(caja1, caja2):
    xA = max(caja1[0], caja2[0])
    yA = max(caja1[1], caja2[1])
    xB = min(caja1[2], caja2[2])
    yB = min(caja1[3], caja2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    caja1Area = (caja1[2] - caja1[0]) * (caja1[3] - caja1[1])
    caja2Area = (caja2[2] - caja2[0]) * (caja2[3] - caja2[1])
    iou = interArea / float(caja1Area + caja2Area - interArea)
    return iou

print(f"Cargando anotaciones desde: {ruta_json_label_studio}")
if not os.path.exists(ruta_json_label_studio):
    print(f"ERROR: El archivo de anotaciones '{ruta_json_label_studio}' no existe. Verifica la ruta.")
    exit()
with open(ruta_json_label_studio, 'r', encoding='utf-8') as f:
    datos_ls_lista = json.load(f)
dataset_final = []
print(f"Se encontraron {len(datos_ls_lista)} tareas en el archivo JSON de Label Studio.")

for tarea_idx, tarea_ls in enumerate(datos_ls_lista):
    print(f"\nProcesando tarea LS con ID de tarea: {tarea_ls.get('id')} (índice {tarea_idx})")
    nombre_archivo_imagen = None
    if 'data' in tarea_ls and isinstance(tarea_ls['data'], dict) and 'image_url' in tarea_ls['data']:
        parsed_url = urllib.parse.urlparse(tarea_ls['data']['image_url'])
        query_params = urllib.parse.parse_qs(parsed_url.query)
        if 'd' in query_params:
            ruta_decodificada_completa = urllib.parse.unquote(query_params['d'][0])
            nombre_archivo_imagen = os.path.basename(ruta_decodificada_completa)
        elif parsed_url.path and parsed_url.path != "/":
             nombre_archivo_imagen = os.path.basename(parsed_url.path)
    if not nombre_archivo_imagen:
        print(f"  ADVERTENCIA: No se pudo extraer el nombre de la imagen para la tarea LS con id {tarea_ls.get('id')}. Verificando claves: data={str(tarea_ls.get('data'))[:200]}. Saltando.")
        continue
    print(f"  Nombre de imagen extraído: {nombre_archivo_imagen}")
    ruta_completa_imagen_fisica = os.path.join(carpeta_base_imagenes_fisicas, nombre_archivo_imagen)
    if not os.path.exists(ruta_completa_imagen_fisica):
        print(f"  ADVERTENCIA: Imagen física no encontrada en {ruta_completa_imagen_fisica}. Saltando.")
        continue
    palabras_ocr, coordenadas_ocr_pixeles_raw = extraer_palabras_y_cajas(
        ruta_completa_imagen_fisica, 
        tesseract_cmd_path=ruta_tesseract_exe, 
        lang='spa'
    )
    if not palabras_ocr:
        print(f"  ADVERTENCIA: No se pudo extraer texto OCR de {nombre_archivo_imagen}. Saltando.")
        continue
    img_pil = Image.open(ruta_completa_imagen_fisica)
    img_width, img_height = img_pil.size
    etiquetas_iob2_para_palabras_ocr = ['O'] * len(palabras_ocr)
    caja_ls_asignada_a_palabra_ocr = [None] * len(palabras_ocr)
    if tarea_ls.get('annotations') and len(tarea_ls['annotations']) > 0 and tarea_ls['annotations'][0].get('result'):
        anotaciones_rectangulos_ls = tarea_ls['annotations'][0]['result']
        for anot_ls in anotaciones_rectangulos_ls:
            if anot_ls.get('type') == 'rectanglelabels' and anot_ls.get('value', {}).get('rectanglelabels'):
                etiqueta_base_ls_str = anot_ls['value']['rectanglelabels'][0]
                if etiqueta_base_ls_str not in BASE_LABELS:
                    print(f"    ADVERTENCIA: Etiqueta LS '{etiqueta_base_ls_str}' no está en BASE_LABELS. Se ignorará para {nombre_archivo_imagen}.")
                    continue
                anot_original_width = anot_ls.get("original_width", img_width)
                anot_original_height = anot_ls.get("original_height", img_height)
                caja_anotacion_pixeles_ls = convertir_coordenadas_ls_a_pixeles(anot_ls['value'], anot_original_width, anot_original_height)
                id_caja_ls = anot_ls.get("id", "unknown_id") 
                for i, palabra_ocr_box_px in enumerate(coordenadas_ocr_pixeles_raw):
                    if calcular_iou(palabra_ocr_box_px, caja_anotacion_pixeles_ls) > 0.3: 
                        if etiquetas_iob2_para_palabras_ocr[i] == 'O':
                             etiquetas_iob2_para_palabras_ocr[i] = etiqueta_base_ls_str 
                             caja_ls_asignada_a_palabra_ocr[i] = id_caja_ls
        for i in range(len(palabras_ocr)):
            etiqueta_base_actual = etiquetas_iob2_para_palabras_ocr[i]
            if etiqueta_base_actual != 'O':
                id_caja_ls_actual = caja_ls_asignada_a_palabra_ocr[i]
                if i > 0 and \
                   etiquetas_iob2_para_palabras_ocr[i-1] == etiqueta_base_actual and \
                   caja_ls_asignada_a_palabra_ocr[i-1] == id_caja_ls_actual:
                    etiquetas_iob2_para_palabras_ocr[i] = f"I-{etiqueta_base_actual}"
                else:
                    etiquetas_iob2_para_palabras_ocr[i] = f"B-{etiqueta_base_actual}"
    etiquetas_ner_ids = []
    for tag_str in etiquetas_iob2_para_palabras_ocr:
        if tag_str not in label2id:
            print(f"  ADVERTENCIA: Etiqueta final '{tag_str}' no encontrada en label2id. Se usará 'O'. Factura: {nombre_archivo_imagen}")
            etiquetas_ner_ids.append(label2id['O'])
        else:
            etiquetas_ner_ids.append(label2id[tag_str])
    coordenadas_ocr_normalizadas_1000 = [normalizar_coordenadas_ocr_a_1000(box, img_width, img_height) for box in coordenadas_ocr_pixeles_raw]

    # **** LÍNEA CLAVE A ASEGURAR / AÑADIR ****
    dataset_final.append({
        "id": str(tarea_ls.get("inner_id", tarea_ls.get("id", os.path.splitext(nombre_archivo_imagen)[0]))),
        "words": palabras_ocr,
        "bboxes": coordenadas_ocr_normalizadas_1000,
        "ner_tags": etiquetas_ner_ids,
        "image_path": ruta_completa_imagen_fisica # <--- ESTA LÍNEA ES LA IMPORTANTE
    })
    print(f"  Factura procesada y añadida al dataset: {nombre_archivo_imagen}")

print(f"\nGuardando dataset procesado en: {archivo_salida_dataset}")
# os.makedirs(os.path.dirname(archivo_salida_dataset), exist_ok=True) # Ya no es necesaria si guardamos en la misma carpeta
with open(archivo_salida_dataset, 'w', encoding='utf-8') as f:
    for entrada in dataset_final:
        f.write(json.dumps(entrada, ensure_ascii=False) + '\n')
print(f"\n¡Proceso completado! Dataset listo para fine-tuning en: {archivo_salida_dataset}")
print(f"Total de ejemplos en el dataset: {len(dataset_final)}")
print("\nMapeo de etiquetas a IDs usado (primeras 10 y últimas 5):")
count = 0
temp_list = list(label2id.items())
for label, id_val in temp_list[:10]:
    print(f"  {label}: {id_val}")
    count +=1
if len(temp_list) > 15:
    print("  ...")
    for label, id_val in temp_list[-5:]:
        print(f"  {label}: {id_val}")
        count+=1
print(f"  Total de etiquetas IOB: {len(label_list)}")