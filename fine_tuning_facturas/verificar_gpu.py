import torch

cuda_disponible = torch.cuda.is_available()
print(f"CUDA disponible: {cuda_disponible}")

if cuda_disponible:
    nombre_gpu = torch.cuda.get_device_name(0)
    print(f"Nombre de la GPU: {nombre_gpu}")
    memoria_total_gpu = torch.cuda.get_device_properties(0).total_memory
    print(f"Memoria GPU Total: {memoria_total_gpu / (1024**3):.2f} GB")
else:
    print("CUDA no está disponible. El entrenamiento se realizará en CPU (será mucho más lento).")

print(f"Versión de PyTorch: {torch.__version__}")
if cuda_disponible:
    print(f"Versión de CUDA compilada con PyTorch: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'No info de CUDA en torch.version'}")