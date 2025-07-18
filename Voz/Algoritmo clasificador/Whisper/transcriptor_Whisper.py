import os
import whisper
from pydub import AudioSegment
from rapidfuzz import process
import unicodedata
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# Diccionario de carpetas y etiquetas
carpetas = {
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_cilindrico": "cilindrico",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_gancho": "gancho",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_obliquo": "obliquo",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_pinza": "pinza",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/agarre_pinza_lateral": "pinza lateral",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/otro": "otro",
    "C:/Users/gerar/Desktop/tfg/Control_de_veu/audio_agarre_normalizados/stop": "stop"
}

# Etiquetas sin incluir "otro"
etiquetas = list(set(carpeta for carpeta in carpetas.values() if carpeta != "otro"))

# Cargar modelo Whisper
modelo = whisper.load_model("small")
prompt = "agarre cilíndrico, agarre gancho, agarre obliquo, agarre pinza, pinza lateral, stop"

# Normalización de audio
def normalizar_audio(ruta_audio):
    audio = AudioSegment.from_file(ruta_audio)
    return audio.apply_gain(-audio.max_dBFS)

# Limpieza de texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

# Clasificación de la palabra
def clasificar_palabra(transcripcion, etiquetas):
    transcripcion_limpia = limpiar_texto(transcripcion)
    etiqueta, score, _ = process.extractOne(transcripcion_limpia, etiquetas)
    return etiqueta if score > 80 else "otro"

# Procesar audios
def procesar_audios():
    archivos, transcripciones, etiquetas_reales = [], [], []

    for carpeta, etiqueta_real in carpetas.items():
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.wav'):
                ruta_audio = os.path.join(carpeta, archivo)

                # Normalizar
                audio_normalizado = normalizar_audio(ruta_audio)
                ruta_temp = f"temp_{archivo}"
                audio_normalizado.export(ruta_temp, format="wav")

                # Transcribir
                resultado = modelo.transcribe(
                    ruta_temp, language='es', temperature=0, best_of=5,
                    initial_prompt=prompt
                )
                transcripcion = resultado['text'].strip()
                etiqueta_predicha = clasificar_palabra(transcripcion, etiquetas)

                archivos.append(archivo)
                transcripciones.append(transcripcion)
                etiquetas_reales.append(etiqueta_real)

                os.remove(ruta_temp)

    return archivos, transcripciones, etiquetas_reales

# Ejecutar el procesamiento
archivos, transcripciones, etiquetas_reales = procesar_audios()

# Crear embeddings
modelo_embedding = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X = modelo_embedding.encode(transcripciones)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, etiquetas_reales, test_size=0.2, random_state=42)

# Entrenar modelo
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.0005,
    max_iter=2000,
    alpha=0.0001,
    random_state=42
)
model.fit(X_train, y_train)

# Mostrar predicciones
for archivo, transcripcion, etiqueta_real, etiqueta_predicha in zip(archivos, transcripciones, etiquetas_reales, model.predict(X)):
    print(f"Archivo: {archivo}")
    print(f"Transcripción: {transcripcion}")
    print(f"Etiqueta real: {etiqueta_real} | Etiqueta predicha: {etiqueta_predicha}")
    print("-" * 50)

# Evaluar precisión
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Guardar modelo
joblib.dump(model, "modelo_transcriptor.pkl")
print("Modelo guardado exitosamente.")

# Matriz de confusión por palabra
cm = confusion_matrix(y_test, y_pred, labels=etiquetas + ["otro"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas + ["otro"])
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión por Palabra")
plt.tight_layout()
plt.show()
