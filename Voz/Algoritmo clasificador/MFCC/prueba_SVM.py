import os
import wave
import pyaudio
from pydub import AudioSegment
import noisereduce as nr

#Configuración
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
DURATION = 3
NUM_GRABACIONES = 5

# Carpeta de salida
BASE_DIR = "datos_reales_para_entrenar"
os.makedirs(BASE_DIR, exist_ok=True)

# Normalizar y reducir ruido
def normalizar_y_filtrar(audio_path):
    audio = AudioSegment.from_file(audio_path)
    audio_data = audio.get_array_of_samples()
    audio_reducido = nr.reduce_noise(y=audio_data, sr=audio.frame_rate)
    
    audio_filtrado = AudioSegment(
        audio_reducido.tobytes(), 
        frame_rate=audio.frame_rate, 
        sample_width=audio.sample_width, 
        channels=audio.channels
    )
    return audio_filtrado.apply_gain(-audio_filtrado.max_dBFS)

# Grabar audio
def grabar_audio(nombre_archivo):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

    print(f"\n Grabando {nombre_archivo}...")
    frames = [stream.read(CHUNK) for _ in range(int(SAMPLE_RATE / CHUNK * DURATION))]
    print(" Grabado.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(nombre_archivo, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

# Main
def main():
    etiqueta = input("Introduce la etiqueta (ej: stop, pinza): ").strip().lower()
    output_dir = os.path.join(BASE_DIR, etiqueta.replace(" ", "_"))
    os.makedirs(output_dir, exist_ok=True)

    for i in range(NUM_GRABACIONES):
        raw_path = f"temp_{etiqueta}_{i}.wav"
        grabar_audio(raw_path)

        # Procesar y guardar en carpeta final
        audio_filtrado = normalizar_y_filtrar(raw_path)
        output_path = os.path.join(output_dir, f"{etiqueta}_{i}.wav")
        audio_filtrado.export(output_path, format="wav")
        os.remove(raw_path)

        print(f"Guardado: {output_path}")

if __name__ == "__main__":
    main()
