# Data Augmentation with Genetic Algorithm

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

- python (>=3.9)
- pandas (>=1.5.0)
- opencv-python (>=4.6.0)
- matplotlib (>=3.6.0)
- sdv (>=0.17.1)
- scikit-learn (>=1.1.2)

### Instalación para el usuario
Usando ``pip``, el sistema de gestión de paqutes:

1. Actualizamos la versión de ``pip``:
```
pip install --user --upgrade pip
```
  
2. Una vez actualizado este comando procederemos a instalar la librería ``daua_ga``:
```
pip install daug_ga
```

### Ejemplo de uso

`$ from daug_ga import split_data, RLGeneticAlgorithm, data_augmentation`

Cargamos los datos

```
from sklearn.datasets import load_iris

import pandas as pd

import numpy as np

data = load_iris() 

df = pd.DataFrame(data=np.c_[data['data'],data['target']], columns=data['feature_names']+['target']) 

x_column_names = df.columns[:-1] 

y_column_name = df.columns[-1]
```

Aplicamos la función split_data 
```
split_data_class = split_data(df, train_size=0.7, x_column_names=x_column_names, y_column_name=y_column_name)

train, test = split_data_class.split_train_test()
```

Aplicamos la función RLGeneticAlgorithm 
```
genetic_class = RLGeneticAlgorithm(df=df, tournament_size_per=0.2, max_generations=2, population_size=10, verbose_model=True, train_size=0.75, x_column_names=x_column_names, y_column_name=y_column_name, train=train, test=test) 

last_population, last_population_fitness = genetic_class.run(df=df, cross_prob=1.0, mut_prob=2.0, elitism=False) 

best_individual, best_individual_fitness = last_population[0], last_population_fitness[0]
```

Aplicamos la función daga_augmentation 
```
data_augmentation_func = data_augmentation(df=df, tournament_size_per=0.2, max_generations=3, population_size=10, verbose_model=True, train_size=0.75, x_column_names=x_column_names, y_column_name=y_column_name, train=train, test=test, distribution=best_individual) 

augmented_data = data_augmentation_funcion.GaussianCopula_augmentation() 

data_augmentation_funcion.plot_new_data_vs_original_data(augmented_data)
```
![Alt text](example/iris_density_plot.jpg?raw=true "Density plot")

![Alt text](example/iris_histogram_plot.jpg?raw=true "Distribution plot")


### Links importantes

- Repositorio del código fuente: <https://github.com/anemartinez1/daug_ga>
- Descargar versiones: <https://pypi.org/project/daug_ga/>
- Seguimiento de problemas: <https://github.com/anemartinez1/daug_ga/issues>
