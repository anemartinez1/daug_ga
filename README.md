# daga

**daug_ga** es un acrónimo de **Data Augmentation with Genetic Algorithm**; este es un módulo de Python para el aumento de datos basado en la librería SDV. Este tiene como objetivo obtener la distribución correspondiente a cada variable y así conseguir mejores resultados en la generación de datos sintéticos. Para ello hace uso de un algortimo genético mono objetivo.

El projecto se empezó en 2022 por [Ane Martínez](https://github.com/anemartinez1, "Ane Martínez") y [Elene Astondoa](https://github.com/eleneastondoa, "Elene Astondoa") como trabajo para la universidad.

## Estructura de carpetas
- daug_ga/
	- __ init __.py
	- functions.py
- LICENSE.txt
- setup.py
- setup.cfg
- README.md

## Instalación

### Dependencias

daga requiere:

- python (>=3.9)
- pandas (>=1.5.0)
- opencv-python (>=4.6.0)
- matplotlib (>=3.6.0)
- sdv (>=0.17.1)
- scikit-learn (>=1.1.2)

### Instalación para el usuario
Usando ``pip``:

1. Primero vamos a actualizar pip a la última versión:

  `$ ! pip install --user --upgrade pip`
  
2. Una vez actualizado este comando procederemos a instalar la librería ``daga``:

  `$ pip install daug_ga`
  
### Links importantes

- Repositorio del código fuente: <https://github.com/anemartinez1/daug_ga>
- Descargar versiones: <https://pypi.org/project/daug_ga/>
- Seguimiento de problemas: <https://github.com/anemartinez1/daug_ga/issues>
