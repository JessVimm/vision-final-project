# vision-final-project

## Requisitos

- Python 3.14.0.
- Bibliotecas utilizadas por el proyecto:
  - numpy
  - scipy
  - trimesh
  - matplotlib
  - tkinter

Instalar las dependencias mediante:

```bash
pip install numpy scipy trimesh matplotlib
```

> Tkinter suele venir incluido en la instalación estándar de Python.


## Ejecución del programa

Desde la terminal, ubicarse en la carpeta del proyecto y ejecutar:

```bash
python ProyectoFinal.py
```


## Selección de modelos

Al iniciar el programa se abrirán dos ventanas de selección de archivos.

Pueden utilizarse los modelos incluidos en la carpeta `objetos_prueba`.

1. Seleccionar el primer modelo tridimensional en formato `.obj`.
2. Seleccionar el segundo modelo tridimensional en formato `.obj`.

Una vez seleccionados ambos archivos, el programa ejecutará automáticamente todo el proceso de comparación.

> Dependiendo de los modelos seleccionados, el procesamiento puede tardar algunos minutos.


## Procesamiento realizado

El sistema realiza las siguientes etapas:

1. Carga y limpieza de las mallas 3D.
2. Alineación mediante tensor de inercia.
3. Voxelización de ambos objetos.
4. Normalización a un volumen discreto común de 15,000 vóxeles.
5. Identificación de vóxeles comunes, sobrantes y faltantes.
6. Construcción de la matriz de costos.
7. Aplicación del Algoritmo Húngaro.
8. Cálculo de la medida de disimilitud.
9. Generación de visualizaciones y animaciones.


## Archivos generados

Al finalizar la ejecución se generan automáticamente:

- Un reporte de resultados en formato `.txt`.
- Una animación `.gif` mostrando la transformación.
- Los modelos normalizados.
- Un modelo intermedio de la transformación.
- El modelo final voxelizado.

Todos los archivos se guardan en el directorio de ejecución del programa.


## Nota importante

El cálculo del Algoritmo Húngaro puede tardar varios minutos dependiendo de la complejidad de los modelos seleccionados.

----------------------------
## Estructura del repositorio

```text
vision-final-project/
│
├── animaciones/
├── comparaciones/
├── nubes_de_puntos/
├── objetos_prueba/
├── proceso_transformacion/
├── reportes/
│
├── Codigo_Proyecto_Final.py
├── Presentación_Proyecto_Final_JC.pdf
├── Reporte_Proyecto_Final_CJ.pdf
├── datos_tablas.txt
├── repositorio_remoto.txt
└── README.md
```

### Descripción de las carpetas

| Carpeta | Descripción |
|----------|------------|
| `animaciones/` | Contiene las animaciones generadas durante el proceso de transformación entre objetos tridimensionales. |
| `comparaciones/` | Almacena imágenes y visualizaciones de las comparaciones y clasificaciones realizadas entre pares de objetos. |
| `nubes_de_puntos/` | Incluye las nubes de puntos obtenidas durante el procesamiento y representación de los modelos 3D. |
| `objetos_prueba/` | Contiene los modelos tridimensionales de ejemplo en formato `.obj` utilizados para realizar pruebas y experimentos. |
| `proceso_transformacion/` | Guarda los modelos intermedios y finales generados durante la transformación óptima entre objetos. |
| `reportes/` | Contiene los archivos de resultados generados automáticamente por el programa, incluyendo métricas, estadísticas de comparación, asignaciones dadas por el Algoritmo Húngaro. |

### Descripción de los archivos principales

| Archivo | Descripción |
|----------|------------|
| `Codigo_Proyecto_Final.py` | Implementación principal de la metodología. |
| `Reporte_Proyecto_Final_CJ.pdf` | Documento técnico que describe la metodología, implementación y resultados obtenidos. |
| `Presentación_Proyecto_Final_JC.pdf` | Presentación utilizada para exponer el proyecto. |
| `datos_tablas.txt` | Archivo con datos numéricos utilizados para la elaboración de tablas y análisis de resultados. |
| `repositorio_remoto.txt` | Contiene la referencia al repositorio remoto del proyecto. |
| `README.md` | Documento de descripción general e instrucciones de uso del proyecto. |

