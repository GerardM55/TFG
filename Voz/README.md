Este directorio contiene todos los scripts relacionados con el procesamiento y clasificación de comandos de voz. 
Los archivos están organizados en tres subcarpetas, en función del tipo de preprocesamiento aplicado.
## Algoritmo clasificador
Esta carpeta incluye todos los algoritmos clasificadores entrenados para ser el algoritmo clasificador final. 
Se incluyen 3 subcarpetas divididas en función del algoritmo de preprocesamiento utilizado. Estas 3 carpetas son: 
### MFCC
Este directorio contiene los algoritmos clasificadores entrenados con el preprocesamiento mediane MFCC (Mel-Frequency Cepstral Coefficients ). 
Los algoritmos clasificadores utilizados son el **MLPClassifier, el Random Forest y el SVM**. 
La estructura y funcionamiento de los 3 códigos implementados es la siguiente:
##### Descripción
El programa realiza las siguientes tareas:

* Carga y procesamiento de audios en formato .wav, aplicando:

* Reducción de ruido

* Normalización

* Extracción de características MFCC + deltas

* Clasificación de comandos mediante el algoritmo clasificador seleccionado optimizado con GridSearchCV.

* Evaluación del modelo usando métricas de precisión y matriz de confusión.

* Exportación del modelo entrenado en formato .pkl.

##### Estructura de carpetas esperada
Los audios deben estar organizados por carpetas, una por tipo de agarre o comando. Ejemplo:
```
audio_agarre_normalizados/
├── agarre_cilindrico/
├── agarre_gancho/
├── agarre_obliquo/
├── agarre_pinza/
├── agarre_pinza_lateral/
├── otro/
└── stop/

```
##### Requisitos
Instala las dependencias necesarias con:

```
pip install -r requirements.txt
```
(El archivo requirements.txt debe incluir: librosa, noisereduce, pydub, scikit-learn, matplotlib, joblib, psutil, etc.)

#### Salida
* Precisión del modelo y visualización de la matriz de confusión

* Mejores hiperparámetros encontrados

### Vosk
El directorio siguiente contiene los scripts utilizados para entrenar los algoritmos clasificadores aplicando el transcriptor Vosk. 
Estos algoritmos son: **Linear Regression, MLPClassifier, Random Forest, SVM y XGBoost**. 
La estructura y funcionamiento de los códigos implementados es la siguiente:
#### Descripción
El sistema sigue los siguientes pasos:
* Carga y preprocesamiento de audios

* Normalización de volumen

* Reducción de ruido de fondo

* Transcripción automática usando el modelo de Vosk en español para convertir voz en texto

* Limpieza del texto transcrito mediante la lliminación de acentos y caracteres especiales

* Vectorización del texto

* Entrenamiento del modelo con el algoritmo clasificador seleccionado. 

* Optimización de hiperparámetros con GridSearchCV

* Evaluación del rendimiento
#### Estructura de las carpetas
Los audios están organizados por etiquetas en las siguientes rutas (modificables en el script):
````
Control_de_veu/
├── audio_agarre_normalizados/
│   ├── agarre_cilindrico/
│   ├── agarre_gancho/
│   ├── agarre_obliquo/
│   ├── agarre_pinza/
│   ├── agarre_pinza_lateral/
│   ├── otro/
│   └── stop/
├── vosk-model-es-0.42/
````
#### Requisitos
Asegúrate de tener instaladas las siguientes librerías de Python:
````
pip install numpy pydub noisereduce vosk scikit-learn matplotlib joblib
````
#### Salida
* Precisión impresa en consola.

* Matriz de confusión mostrada con matplotlib.

### Whisper
En esta carpeta se encuentra un modelo con el algoritmo clasificador **MLPCLassifier** y el transcriptor **Whisper ai**. 
La estructura y funcionamiento del códigos es la siguiente:
#### Descripción
El flujo del sistema es el siguiente:

* Carga y preprocesamiento de audios

* Transcripción con el modelo Whisper small en español.

* Limpieza y clasificación semántica

* Extracción de características semánticas

* Clasificación con el modelo MLPClassifier que se entrena con las transcripciones vectorizadas.

* Evaluación mediante el accuracy y una matriz de confusión.

#### Estructura de las carpetas
La estructura de las carpetas es la siguiente: 
````
Control_de_veu/
└── audio_agarre_normalizados/
    ├── agarre_cilindrico/
    ├── agarre_gancho/
    ├── agarre_obliquo/
    ├── agarre_pinza/
    ├── agarre_pinza_lateral/
    ├── stop/
    └── otro/
````
#### Requisitos
Las líbrerias necesarias son las siguientes: 
````
pip install openai-whisper pydub rapidfuzz sentence-transformers scikit-learn matplotlib joblib
````
#### Salida
* Se calcula la precisión del modelo sobre el conjunto de prueba.

* Se muestra una matriz de confusión con los resultados por palabra.

## Detección de voz
En este directorio se encuentran los scripts **deteccion_hyperparametros.py** y **deteccion_voz.py** encargados de la detección de voz del usuario. 
Estos scripts no han sido aplicados al modelo final. A continuación, se detalla los requisitos y estructura de los códigos. 
### deteccion_hyperparametros
Este script implementa un sistema de detección de voz que distingue entre la voz del usuario y la de otras personas. 
Para ello, utiliza un preprocesamiento con MFCC y una red neuronal X-Vector entrenada con PyTorch.

#### Descripción
El sistema realiza las siguientes acciones:
* Carga y preprocesamiento de datos de audio

* Extracción de características (MFCC) con data augmentation

* Definición de un Dataset personalizado

* Entrenamiento de una red neuronal X-Vector

* Búsqueda de hiperparámetros

* Evaluación del rendimiento

* Verificación de hablante a partir de un archivo .wav

#### Estructura de las carpetas
La estructura de las carpetas es la siguiente: 
````
Control_de_veu/
├── audios_deteccio_veu/
│   ├── meva_veu_aug_2/         # Audios de entrenamiento - mi voz
│   ├── altres_veu_aug_2/       # Audios de entrenamiento - otras voces
│   ├── meva_veu_test/          # Audios de test - mi voz
│   └── altres_veu_test/        # Audios de test - otras voces
````
#### Requisitos
Las líbrerias necesarias para ejecutar este script son: 
````
pip install torch librosa numpy scikit-learn
````
#### Salida
* Precisión del modelo. 

* Mejores hiperparámetros encontrados

### deteccion_voz
Este script es una replica del script anterior pero utilizando los mejores hiperparámetros extraidos del código anterior. 
Por lo tanto, la estructura y los requisitos del código son los mismos que el anterior script. 

## Preprocesamiento
En este directorio se encuentran los scripts utilizados para el preprocesamiento de los audios. 
Estos basicamente se dividen en los scripts de data augmentation de la detección de voz, donde se encuentran 
el script **DA_deteccion_mi_voz.py** y **DA_deteccion_otra_voz.py**. Como el nombre del archivo indica,
uno es para el DA de los audios de la voz del usuario y el otro para el DA de los audios externos. 
Posteriormente, hay el script **data_augmentation_audios.py** encargado del DA de los audios de los agarres 
y finalmente, el script **normalizacion_audio.py**, que es el script referente a la normalización de los audios a 16 KHz. 

### DA_deteccion_mi_voz.py, DA_deteccion_otra_voz.py y data_augmentation_audios.py
Estos tres scripts son los encargados del DA. La diferencia entre ellos radica en las carpetas asociadas.

#### Descripción
Los scripts realizan el siguiente: 
* Cargar audios `.wav` desde la carpeta de origen
* Aplicar **data augmentation**
* Guardar tanto el audio original como las versiones aumentadas en la carpeta de destino
  
#### Técnicas aplicadas
* **PitchShift**: Cambia el tono del audio
* **TimeStretch**: Acelera o ralentiza ligeramente el audio
* **AddGaussianNoise**: Añade ruido gaussiano al audio
* **Gain**: Ajusta el volumen general

#### Estructura de las carpetas
La estructua de las carpetas para el script **DA_deteccion_mi_voz.py** es la siguiente: 
````
Control_de_veu/
│
├── audios_deteccio_veu/
│   ├── meva_veu_aug/
│   ├── meva_veu_aug_2/
````
Para el script **DA_deteccion_otra_voz.py**: 
````
Control_de_veu/
│
├── audios_deteccio_veu/
│   ├── altres_veu_aug/
│   ├── altres_veu_aug_2/
````
Y para el script **data_augmentation_audios.py**: 
````
Control_de_veu/
│
└── audio_agarre/
    ├── agarre_X/
    └── agarre_X_aug/
````
#### Requisitos
Las líbrerias necesarias son: 
````
pip install librosa audiomentations numpy soundfile
````
#### Salida
La salida de los scripts es el audio original junto con los audios generados aplicando DA. 

### normalizacion_audio.py
Este script normaliza los audios de entrada a una frecuencia de 16 KHz.
#### Descripción
* Recorre de forma recursiva una carpeta de entrada con archivos .wav

* Convierte los audios que no estén a 16 kHz

* Guarda los audios normalizados en una carpeta de salida

#### Estructura de las carpetas
La estructura de las carpetas es la siguiente. 
````
Control_de_veu/
│
└── audio_agarre/
    ├── agarre_X/
    └── agarre_X_aug/
````
#### Requisitos
Las líbrerias necesarias son: 
````
pip install librosa soundfile
````
#### Salida
La salida del script son los audios normalizados a 16 KHz. 
