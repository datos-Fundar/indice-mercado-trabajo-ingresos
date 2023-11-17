# Índice de Género, Trabajo e Ingresos en Argentina

## Descripción del Repositorio

Este repositorio contiene los archivos necesarios para el cálculo del [Índice de Género, Trabajo e Ingresos](https://fund.ar/publicacion/indice-de-genero-trabajo-e-ingresos/) desarrollado por [Fundar](https://fund.ar/). El índice, creado por primera vez, sistematiza la desigualdad económica de género a nivel provincial en Argentina, revelando las heterogeneidades entre las 24 jurisdicciones para trazar un perfil único de cada una.

## Estructura del Repositorio

- **basemaps:** Contiene archivos relacionados con los aglomerados de la Encuesta Permanente de Hogares (EPH).
- **data_input:** Directorio para almacenar datos de entrada necesarios para los cálculos.
- **data_output:**
  - **componentes_e_indice:** Resultados intermedios y el índice final.
  - **indicadores:** Archivos CSV con indicadores utilizados en el cálculo del índice.
  - **indicadores_auxiliares:** Archivos CSV con indicadores auxiliares.
- **docs:** Documentación relacionada con el proyecto.
- **figs:** Gráficos y visualizaciones generadas durante el análisis.
- **modulos:**
  - **calculos:** Notebooks de Jupyter para realizar cálculos específicos del índice.
  - **diccionarios:** Notebooks y archivos pickle que contienen diccionarios utilizados en el proceso.
  - **funciones:** Notebooks con funciones específicas para el proyecto.
  - **visualizacion:** Notebooks para la visualización de datos y la creación de gráficos.

## Instrucciones de Uso

1. **Datos de Entrada:** Coloque los datos de entrada en el directorio `data_input`.
2. **Cálculos del Índice:** Ejecute los notebooks en el directorio `modulos/calculos` en el orden especificado para calcular el índice.
3. **Visualización:** Explore los notebooks en `modulos/visualizacion` para visualizar los resultados y generar gráficos.
4. **Documentación Adicional:** Consulte la carpeta `docs` para obtener información adicional sobre el proyecto.

