# Proyecto de Clasificación de Imágenes con Momentos de Hu y Zernike

Este proyecto tiene como objetivo la clasificación de formas geométricas (círculo, triángulo y cuadrado) a partir de imágenes proporcionadas por el usuario, utilizando características extraídas mediante los Momentos de Hu y Zernike. El sistema está diseñado para procesar imágenes dibujadas a mano por el usuario, realizar predicciones de clasificación y comparar los resultados de ambos métodos.

## Requisitos

- **C++** (para la implementación de los métodos de procesamiento de imágenes y cálculos de Momentos de Hu y Zernike).
- **Java** (para la implementación de la interfaz móvil y la comunicación con el backend).
- **OpenCV** (para el procesamiento de imágenes y cálculo de Momentos de Hu y Zernike).
- **Android Studio** (para la interfaz móvil).
- **GCC/G++** (para compilar el código en C++).

## Descripción del Proyecto

El sistema permite al usuario cargar una imagen dibujada, que luego es procesada por un conjunto de algoritmos en C++ para extraer los Momentos de Hu y Zernike. Las características extraídas se utilizan para clasificar la forma de la imagen en una de las tres categorías: Círculo, Triángulo o Cuadrado. El sistema ofrece dos métodos de predicción: **Momentos de Hu** y **Momentos de Zernike**.

### Flujo del Proyecto

1. El usuario dibuja una figura en la interfaz móvil.
2. La imagen es enviada al servidor en C++ donde se procesan los Momentos de Hu y Zernike.
3. El sistema clasifica la imagen en una de las tres categorías.
4. Los resultados de la clasificación se muestran al usuario en la interfaz móvil.
