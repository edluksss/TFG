# Trabajo de Fin de Grado (TFG): Detección y clasificación morfológica de nebulosas planetarias en imágenes astrofísicas mediante aprendizaje automático

Este repositorio contiene el código y los recursos para mi Trabajo de Fin de Grado (TFG), que se centra en la segmentación y clasificación morfológica de imágenes astrofísicas de nebulosas planetarias. Hasta el momento, he realizado parte de la segmentación de las imágenes.

## Estructura del Repositorio

El repositorio está organizado de la siguiente manera:

- `config`: Contiene archivos de configuración como `environment.yml`, `python_version.txt` y `requirements.txt`.
- `doc`: Contiene documentación adicional y hojas de ruta del proyecto.
- `pnebulae_torch`: Contiene el código fuente del proyecto, incluyendo módulos para la gestión de conjuntos de datos, normalización, preprocesamiento y utilidades. Gracias a este directorio ya se puede instalar el código fuente de mi proyecto como una librería de Python ejecutando en una terminal el siguiente comando:
```bash
pip install git+https://github.com/edluksss/TFG.git
```
- `res`: Contiene los recursos utilizados en el proyecto, como el catálogo de imágenes astrofísicas junto a sus máscaras binarias, así como información adicional sobre el conjunto de datos (etiquetas sobre su morfología y sobre el conjunto al que van a pertenecer durante todo el proyecto).
- `test`: Contiene scripts y cuadernos Jupyter para pruebas y visualizaciones. En este directorio se encuentran todos los resultados obtenidos con la segmentación no supervisada y parte de la segmentación supervisada. Para las pruebas de segmentación supervisada con arquitecturas más sotisficadas como UNet, FPN... se pueden ver los resultados en el siguiente enlace de WandB (Weights and Biases):
[Proyecto WandB](https://wandb.ai/edluksss_org/segmentation_TFG?nw=nwuseredluksss)

## Cómo Instalar

Para instalar las dependencias necesarias para este proyecto, puedes utilizar el archivo `requirements.txt` o `environment.yml` en la carpeta `config`.

Además, se recomienda instalar la misma librería del repositorio como se ha indicado anteriormente.

