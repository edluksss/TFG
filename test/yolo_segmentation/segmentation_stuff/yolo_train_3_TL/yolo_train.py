from ultralytics import YOLO, checks, hub
checks()

hub.login('71454859f9a2ce11ea390ba04b99fac877ea55f8f7')

model = YOLO('https://hub.ultralytics.com/models/RCZcB6lDhfLyeSEjuuXV')
results = model.train()











import os
import shutil


# Cambiamos el directorio de trabajo
os.chdir(os.environ["HOME"])

# Eliminamos la carpeta datasets para que pueda continuar el siguiente entrenamiento
carpeta = 'datasets/'

# Verifica si la carpeta existe
if os.path.exists(carpeta):
    # Elimina la carpeta y todo su contenido
    shutil.rmtree(carpeta)
    print(f'La carpeta {carpeta} ha sido eliminada.')
else:
    print(f'La carpeta {carpeta} no existe.')