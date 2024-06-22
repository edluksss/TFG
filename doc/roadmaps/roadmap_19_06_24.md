# Roadmap de ideas a realizar - 19/06/2024
Este documento se realiza para reflejar las ideas que comentamos en la reunión del día 19 de junio de 2024, desarrollarlas como yo las he entendido y como las voy a realizar por si alguna cosa no hubiese quedado del todo clara y necesite corrección.

## 1. Técnica de Early Stop para incluir clústers como fondo
Incluir un método con el que, en vez de elegir manualmente un porcentaje de píxeles máximos de fondo que las segmentaciones pueden tener, considerar iterativamente clústers como fondo hasta cumplir una condición de comparación de diferencia de contraste entre los clústers.

En una primera instancia la condición se va a elegir del siguiente modo:
- Se hace un análisis, gracias a las máscaras, de la media y desviación típica de la diferencia de contraste que hay entre píxeles de fondo alrededor de la nebulosa (por ejemplo separados 5 píxeles de la nebulosa) y píxeles de dentro del contorno de la nebulosa (por ejemplo 5 píxeles hacía dentro).

- Una vez tenemos estas medidas vamos a utilizarlas como guía para la consideración de cluster como fondo iterativamente (parecido a lo que ya está implementado)
    - 1er cluster se considera como fondo
    - Siguiente cluster se calcula la diferencia de contraste entre x píxeles hacia el interior y x hacia el exterior. 
    - Si la diferencia entra dentro de unos umbrales definidos (mediante el análisis anterior) se para de procesar clusters, sino se sigue procesando el siguiente.

- Observamos las diferentes segmentaciones que se han realizado y la seleccionada finalmente (igual que como se hace ahora)

Otra idea a considerar quizás puede ser la de procesar todas las segmentaciones y guardar los valores de diferencia de contraste y quedarnos con la que sea la mayor diferencia, siempre que la parte del interior del cluster no corresponda con el valor máximo de la imagen (ya que en ese caso la gran diferencia de contraste se debería a a que la estrella principal de la nebulosa).

