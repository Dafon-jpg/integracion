from pdf2image import convert_from_path
from PIL import Image # PIL es parte de Pillow, pdf2image lo necesita
import os
import glob # Para buscar archivos que coincidan con un patrón

def convertir_pdfs_en_lote(carpeta_entrada_pdfs, carpeta_salida_imagenes, poppler_path=None):
    """
    Convierte la primera página de todos los archivos PDF en una carpeta a imágenes PNG.

    Args:
        carpeta_entrada_pdfs (str): La ruta completa a la carpeta que contiene los archivos PDF.
        carpeta_salida_imagenes (str): La carpeta donde se guardarán las imágenes PNG.
        poppler_path (str, optional): La ruta a la carpeta 'bin' de Poppler (para Windows).
    """
    try:
        # Asegurarse de que la carpeta de salida exista
        if not os.path.exists(carpeta_salida_imagenes):
            os.makedirs(carpeta_salida_imagenes)
            print(f"Carpeta de salida creada: {carpeta_salida_imagenes}")

        # Buscar todos los archivos .pdf en la carpeta de entrada
        # Usamos os.path.join para construir la ruta correctamente y glob para encontrar los PDFs
        patron_busqueda_pdfs = os.path.join(carpeta_entrada_pdfs, "*.pdf")
        lista_archivos_pdf = glob.glob(patron_busqueda_pdfs)

        if not lista_archivos_pdf:
            print(f"No se encontraron archivos PDF en: {carpeta_entrada_pdfs}")
            return

        print(f"Se encontraron {len(lista_archivos_pdf)} archivos PDF para convertir.")
        print(f"Poppler path a utilizar: {poppler_path if poppler_path else 'No especificado (se buscará en PATH)'}")

        archivos_convertidos = 0
        archivos_fallidos = 0

        for i, ruta_pdf in enumerate(lista_archivos_pdf):
            nombre_archivo_pdf = os.path.basename(ruta_pdf)
            nombre_base_sin_extension = os.path.splitext(nombre_archivo_pdf)[0]
            nombre_imagen_salida = f"{nombre_base_sin_extension}.png" # Mismo nombre, extensión PNG
            ruta_completa_imagen_salida = os.path.join(carpeta_salida_imagenes, nombre_imagen_salida)

            print(f"\n[{i+1}/{len(lista_archivos_pdf)}] Procesando: {nombre_archivo_pdf}")

            try:
                # Convertir PDF a una lista de objetos Image
                if poppler_path:
                    images = convert_from_path(ruta_pdf, 
                                               first_page=1, 
                                               last_page=1, 
                                               poppler_path=poppler_path,
                                               fmt='png', # Especificar formato de salida
                                               thread_count=2) # Puedes ajustar thread_count
                else:
                    images = convert_from_path(ruta_pdf, 
                                               first_page=1, 
                                               last_page=1,
                                               fmt='png',
                                               thread_count=2)
                
                if images:
                    images[0].save(ruta_completa_imagen_salida, "PNG")
                    print(f"  Éxito: Imagen guardada en: {ruta_completa_imagen_salida}")
                    archivos_convertidos += 1
                else:
                    print(f"  Error: No se pudieron generar imágenes desde {nombre_archivo_pdf}.")
                    archivos_fallidos += 1

            except Exception as e:
                print(f"  Error al convertir {nombre_archivo_pdf}: {e}")
                if "Unable to get page count" in str(e) or "Poppler" in str(e) and poppler_path is None:
                    print("  *** POSIBLE PROBLEMA CON POPPLER ***")
                archivos_fallidos += 1
        
        print(f"\n--- Resumen de Conversión ---")
        print(f"Total de PDFs procesados: {len(lista_archivos_pdf)}")
        print(f"Imágenes generadas con éxito: {archivos_convertidos}")
        print(f"Conversiones fallidas: {archivos_fallidos}")


    except Exception as e:
        print(f"Ocurrió un error general durante el proceso en lote:")
        print(e)

# --- CONFIGURA ESTO ---
# 1. Pon aquí la ruta COMPLETA a la CARPETA que contiene tus 50 archivos PDF.
#    Ejemplo para Windows: ruta_carpeta_con_mis_pdfs = r"C:\Users\TuUsuario\Desktop\Facturas_PDF_para_Convertir"
ruta_carpeta_con_mis_pdfs = r"C:\Users\Usuario\Desktop\Facturas para Fine-Tuning\Primeras 50" 

# 2. Define dónde quieres guardar las imágenes (dentro de tu carpeta fine_tuning_facturas).
#    Esta es la carpeta que configuraste en Label Studio como "Absolute local path"
#    para el "Source Storage".
carpeta_salida_final_imagenes = r"C:\Users\Usuario\Renombrador de archivos\imagenes_facturas" # Ajusta si es diferente

# 3. SOLO PARA WINDOWS: Si Poppler no está en tu PATH, especifica la ruta a la carpeta 'bin' de Poppler.
#    Ejemplo: ruta_poppler_bin = r"C:\Program Files\poppler-24.02.0\Library\bin"
ruta_poppler_bin = r"C:\plopper\poppler-24.08.0\Library\bin" # La ruta que me diste antes, ¡verifica que sea la correcta!

# --- EJECUTAR LA CONVERSIÓN EN LOTE ---
if ruta_carpeta_con_mis_pdfs == r"C:\plopper\poppler-24.08.0\Library\bin":
    print("Por favor, edita la variable 'ruta_carpeta_con_mis_pdfs' en el script con la ruta a tu carpeta de PDFs.")
else:
    if not os.path.isdir(ruta_carpeta_con_mis_pdfs):
        print(f"Error: La carpeta de entrada de PDFs no existe o no es una carpeta: {ruta_carpeta_con_mis_pdfs}")
    else:
        convertir_pdfs_en_lote(ruta_carpeta_con_mis_pdfs, carpeta_salida_final_imagenes, poppler_path=ruta_poppler_bin)