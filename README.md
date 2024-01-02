# Índice de Género, Trabajo e Ingresos en Argentina

## Descripción del Repositorio

La desigualdad de género ha adquirido una visibilidad pública sin precedentes en los últimos años, convirtiéndose en un elemento clave en la planificación del desarrollo del país. Reconociendo la complejidad estructural de este fenómeno y su desafío para la política pública, desde [Fundar](https://fund.ar/) elaboramos un Índice Subnacional de Igualdad de Género (ISIG).

El ISIG aborda la desigualdad de género en las 24 jurisdicciones del país y consta de cuatro aspectos: 
- **Decisión y representación política** 
- **Trabajo e ingresos**
- **Educación y oportunidades**
- **Salud y protección**
Cada medida del ISIG utiliza indicadores construidos de manera sintética y agregada para ofrecer una comprensión integral de la situación de género en cada provincia.

Este repositorio contiene los archivos necesarios para el cálculo de la segunda edición del ISIG: el [Índice de Género, Trabajo e Ingresos](https://fund.ar/publicacion/indice-de-genero-trabajo-e-ingresos/) (IGTI). El IGTI, creado por primera vez, sistematiza la desigualdad económica de género a nivel provincial en Argentina, revelando las heterogeneidades entre las 24 jurisdicciones para trazar un perfil único de cada una.

## Datos del índice agregado 

La arquitectura de este índice se basa en 7 [indicadores](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos#indicadores) agrupados en dos [componentes](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos?tab=readme-ov-file#componentes). Estos dos componentes luego se agregan para constituir el [Índice de Género, Trabajo e Ingresos (IGTI)](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos?tab=readme-ov-file#%C3%ADndice-de-g%C3%A9nero-trabajo-e-ingresos), que está disponible para cada una de las 24 jurisdicciones de Argentina.

Los datos agregados y procesados de los indicadores y componentes del IGTI se definen de la siguiente manera y se incluyen en enlaces.

### Indicadores
- La Actividad (A) se define como el cociente entre la tasa de actividad de mujeres y la tasa de actividad de varones, representado por la ecuación: $$\text{A} = \dfrac{\text{Tasa de actividad de mujeres}}{\text{Tasa de actividad de varones}}$$ Para acceder a las tasas de actividad correspondientes para varones y mujeres, junto con los ratios para las 24 jurisdicciones de Argentina consultar [este enlace](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/blob/main/data_output/indicadores/01_ratio_actividad.csv)
- La variable Formalidad (F) se define como el cociente entre la tasa de mujeres asalariadas con descuento jubilatorio y la tasa de varones asalariadas con descuento jubilatorio, representado por la ecuación: $$\text{F} = \dfrac{\text{Tasa de mujeres asalariadas con dto jubilatorio}}{\text{Tasa de varones asalariadas con dto jubilatorio}}$$ Para acceder a las tasas de F de varones y mujeres, junto con los ratios para las 24 jurisdicciones de Argentina consultar [este enlace](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/blob/main/data_output/indicadores/02_ratio_formalidad.csv)
- La Jornada laboral (JL) se define como el cociente entre las horas semanales promedio remuneradas de mujeres y varones, trabajadas en la ocupación principal, representado por la ecuación: $$\text{JL} = \dfrac{\text{Horas promedio trabajadas de mujeres}}{\text{Horas promedio trabajadas de varones}}$$ Para acceder a los valores de JL correspondientes para varones y mujeres, junto con los ratios para las 24 jurisdicciones de Argentina consultar [este enlace](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/blob/main/data_output/indicadores/03_ratio_jornada_laboral.csv)
- El Ingreso salarial (IS) se define como el cociente entre el ingreso laboral promedio de la ocupación principal de mujeres y varones, representado por la ecuación: $$\text{IS} = \dfrac{\text{Ingreso laboral promedio de la ocupación principal}}{\text{Ingreso laboral promedio de la ocupación principal de varones}}$$ Para acceder a los valores de IS de varones y mujeres, junto con los ratios para las 24 jurisdicciones de Argentina consultar [este enlace](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/blob/main/data_output/indicadores/04_ratio_ingreso_salarial.csv)

- La Jornada no paga (JNP) se define como el cociente entre los minutos promedio de trabajo no remunerados de mujeres y varones, representado por la ecuación: $$\text{JNP} = \dfrac{\text{Minutos promedio de mujeres}}{\text{Minutos promedio de varones}}$$ Para acceder a los valores de JNP correspondientes para varones y mujeres, junto con los ratios para las 24 jurisdicciones de Argentina consultar [este enlace](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/blob/main/data_output/indicadores/05_ratio_jornada_no_paga.csv)
- Los Ingresos propios en población inactiva (IPI) se define como el cociente entre la tasa de mujeres con ingresos propios dentro del total de inactivas que no estudia y la de varones, representado por la ecuación: $$\text{IPI} = \dfrac{\text{Tasa de mujeres con ingresos propios dentro del total de inactivas que no estudia}}{\text{Tasa de varones con ingresos propios dentro del total de inactivos que no estudian}}$$ Para acceder a los valores de IPI correspondientes para varones y mujeres, junto con los ratios para las 24 jurisdicciones de Argentina consultar [este enlace](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/blob/main/data_output/indicadores/06_ratio_inactivos_con_ingreso.csv)
- La No-pobreza en hogares con menores de 25 años según jefatura (NP) se define como el cociente entre la proporción de hogares no pobres con jefe femenino y la proporción de hogares no pobres con jefe masculino, representado por la ecuación: $$\text{NP} = \dfrac{\text{Proporción de hogares no pobres con jefe femenino}}{\text{Proporción de hogares no pobres con jefe masculino}}$$ Para acceder a los valores de NP correspondientes para varones y mujeres, junto con los ratios para las 24 jurisdicciones de Argentina consultar [este enlace](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/blob/main/data_output/indicadores/07_ratio_hogares_no-pobres_jefatura.csv)

### Componentes 
- Inserción laboral $(C_{IL})$: representa el cociente entre la suma de los indicadores de Actividad (A), Formalidad (F), Jornada laboral (JL) e Ingreso salarial (IS), dividida por 4, según la ecuación: $$\text{Insercion laboral }(C_{IL}) = \dfrac{(A + F + JL + IS)}{4}$$ Para acceder a los ratios correspondientes, consultar [este enlace](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/blob/main/data_output/componentes_e_indice/01_insercion_laboral.csv)
- Uso del tiempo y oportunidades $(C_{UTO})$: representa el cociente entre la suma de Jornada no paga (JNP), Ingresos propios en población inactiva (IPI) y No-pobreza en hogares con menores de 25 años según jefatura (NP), dividida por 3, según la ecuación: $$\text{Uso del tiempo y oportunidades }(C_{UTO}) = \dfrac{(JNP + IPI + NP)}{3}$$ Para acceder a los ratios correspondientes, consultar [este enlace](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/blob/main/data_output/componentes_e_indice/02_uso_del_tiempo_y_oportunidades.csv)

### Índice de Género, Trabajo e Ingresos
- El IGTI se define como la raíz cuadrada del producto de Insercion laboral $(C_{IL})$ y Uso del tiempo y oportunidades $(C_{UTO})$, según la ecuación:  $$\text{IGTI} = \sqrt{(C_{IL}*C_{UTO})}$$ Consultar [aquí](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/blob/main/data_output/componentes_e_indice/01_indice_GTI.csv) para acceder a los datos del IGTI

## Estructura del Repositorio

- **[`basemaps`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/basemaps):** Contiene archivos relacionados con los aglomerados de la Encuesta Permanente de Hogares (EPH).
- **[`data_input`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/data_input):** Almacena datos de entrada necesarios para los cálculos.
- **[`data_output`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/data_output):**
  - **[`componentes_e_indice`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/data_output/componentes_e_indice):** Resultados intermedios y el índice final.
  - **[`indicadores`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/data_output/indicadores):** Archivos .csv con indicadores utilizados en el cálculo del índice.
  - **[`indicadores_auxiliares`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/data_output/indicadores_auxiliares):** Archivos .csv con indicadores auxiliares.
- **[`docs`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/docs):** Documentación relacionada con el proyecto.
- **[`figs`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/figs):** Gráficos y visualizaciones generadas durante el análisis.
- **[`modulos`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/modulos):**
  - **[`calculos`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/modulos/calculos):** Juypter notebooks para realizar cálculos específicos del índice.
  - **[`diccionarios`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/modulos/diccionarios):** Juypter notebooks y archivos pickle que contienen diccionarios utilizados en el proceso.
  - **[`funciones`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/modulos/funciones):** Juypter notebooks con funciones específicas para el proyecto.
  - **[`visualizacion`](https://github.com/datos-Fundar/indice-mercado-trabajo-ingresos/tree/main/modulos/visualizacion):** Juypter notebooks para la visualización de datos y la creación de gráficos.


## Instrucciones de Uso

1. **Datos de Entrada:** Utilice los datos de entrada en el directorio `data_input`.
2. **Cálculos del Índice:** Ejecute los notebooks en el directorio `modulos/calculos` en el orden especificado para calcular el índice.
3. **Visualización:** Explore los notebooks en `modulos/visualizacion` para visualizar los resultados y generar gráficos.
4. **Documentación Adicional:** Consulte la carpeta `docs` para obtener información adicional sobre el proyecto.

## Recursos adicionales

- [Fichas provinciales Índice de Género, Trabajo e Ingresos](https://fund.ar/wp-content/uploads/2023/11/Fundar_Indice-Genero-Trabajo-Ingresos_Fichas_Provinciales_CC-BY-NC-ND-4.0.pdf)
- [Documento completo Índice de Género, Trabajo e Ingresos](https://fund.ar/wp-content/uploads/2023/11/Fundar_Indice-Genero-Trabajo-Ingresos_CC-BY-NC-ND-4.0-1.pdf)
- [Fichas provinciales Índice de Género, Decisión y Representación](https://fund.ar/wp-content/uploads/2023/03/FU_Genero_Fichas_Indice_Final-1.pdf)
- [Documento completo Índice de Género, Decisión y Representación](https://fund.ar/publicacion/indice-genero-decision-representacion/)
