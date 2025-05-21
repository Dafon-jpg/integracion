import pytesseract
from PIL import Image
import os

def extraer_palabras_y_cajas(ruta_imagen, tesseract_cmd_path=None, lang='spa'):
    """
    Extrae palabras y sus coordenadas (bounding boxes) de una imagen usando Tesseract OCR.

    Args:
        ruta_imagen (str): La ruta completa al archivo de imagen.
        tesseract_cmd_path (str, optional): La ruta al ejecutable de tesseract.
                                          Ej: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                                          Si es None, intentará usar la variable de entorno PATH.
        lang (str, optional): El idioma para el OCR (ej. 'spa' para español).

    Returns:
        tuple: (list_de_palabras, list_de_coordenadas) o (None, None) si falla.
    """
    try:
        if tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

        print(f"Procesando imagen: {ruta_imagen}")
        print(f"Usando Tesseract desde: {pytesseract.pytesseract.tesseract_cmd}")
        print(f"Idioma para OCR: {lang}")

        img = Image.open(ruta_imagen)

        # Usar image_to_data para obtener información detallada, incluyendo coordenadas
        # output_type.DICT devuelve un diccionario con claves como 'text', 'left', 'top', 'width', 'height', 'conf'
        ocr_result = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

        palabras = []
        coordenadas = [] # Formato: [x_inicio, y_inicio, x_fin, y_fin]

        n_boxes = len(ocr_result['level']) # Número de cajas detectadas (palabras, líneas, etc.)
        for i in range(n_boxes):
            # Tomamos solo las palabras (level 5) y con una confianza razonable
            if ocr_result['level'][i] == 5: # Nivel 5 corresponde a palabra
                texto_palabra = ocr_result['text'][i].strip()
                confianza = int(float(ocr_result['conf'][i])) # Convertir confianza a entero

                if texto_palabra and confianza > 30: # Puedes ajustar el umbral de confianza (0-100)
                    x = ocr_result['left'][i]
                    y = ocr_result['top'][i]
                    w = ocr_result['width'][i]
                    h = ocr_result['height'][i]

                    palabras.append(texto_palabra)
                    coordenadas.append([x, y, x + w, y + h])

        if palabras:
            print(f"Se extrajeron {len(palabras)} palabras con sus coordenadas.")
            return palabras, coordenadas
        else:
            print("No se pudieron extraer palabras de la imagen con la confianza suficiente.")
            return None, None

    except FileNotFoundError:
        print(f"Error: No se encontró la imagen en la ruta: {ruta_imagen}")
        return None, None
    except pytesseract.TesseractNotFoundError:
        print("\n*** ERROR: TESSERACT NO ENCONTRADO ***")
        print("Asegúrate de que Tesseract OCR esté instalado y en el PATH del sistema,")
        print("o proporciona la 'tesseract_cmd_path' correcta en la función.")
        print("Ej. para Windows: r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
        return None, None
    except Exception as e:
        print(f"Ocurrió un error extrayendo texto y cajas: {e}")
        return None, None

# --- CONFIGURA ESTO ---
# 1. Esta debería ser la ruta a la imagen que generaste en el paso anterior.
ruta_mi_imagen_factura = os.path.join(".", "imagenes_facturas", "factura_para_anotar_01.png")

# 2. SOLO SI TESSERACT NO ESTÁ EN TU PATH: Especifica la ruta al ejecutable tesseract.exe
#    Descomenta y edita la siguiente línea si es necesario.
#    Ejemplo Windows: ruta_tesseract_exe = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#    En macOS/Linux, si lo instalaste con brew/apt, normalmente no necesitas esto (déjalo en None).
ruta_tesseract_exe = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # Cambia esto si es necesario en Windows

# 3. Idioma de tus facturas (por defecto 'spa' para español)
idioma_ocr = 'spa'

# --- EJECUTAR LA EXTRACCIÓN ---
if not os.path.exists(ruta_mi_imagen_factura):
    print(f"La imagen de factura no se encontró en: {ruta_mi_imagen_factura}")
    print("Asegúrate de haber ejecutado 'convertir_pdf.py' primero y que la imagen exista.")
else:
    palabras_extraidas, coordenadas_extraidas = extraer_palabras_y_cajas(
        ruta_mi_imagen_factura, 
        tesseract_cmd_path=ruta_tesseract_exe,
        lang=idioma_ocr
    )

    if palabras_extraidas and coordenadas_extraidas:
        print("\n--- PALABRAS EXTRAÍDAS ---")
        for i, palabra in enumerate(palabras_extraidas):
            print(f"Palabra: '{palabra}', Coordenadas: {coordenadas_extraidas[i]}")

        # Opcional: Guardar en un archivo para revisarlo más fácil
        with open(os.path.join(".", "imagenes_facturas", "factura_01_palabras_cajas.txt"), "w", encoding="utf-8") as f:
            for i, palabra in enumerate(palabras_extraidas):
                f.write(f"Palabra: {palabra}\tCaja: {coordenadas_extraidas[i]}\n")
            print(f"\nResultados también guardados en: {os.path.join('.', 'imagenes_facturas', 'factura_01_palabras_cajas.txt')}")