# Importar librerías
import numpy as np
from tensorflow.python.keras.models import load_model

import tensorflow as tf
import pyaudio
import wave
import scipy.fftpack as fourier
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import time
from tkinter import *
from tkinter import ttk

Etiquetas = ['DO', 'RE', 'MI', 'FA', 'SOL', 'LA', 'SI']
Modelo_Importado = load_model('Mi_RNA')
Audio_encontrado = ""


def record():
    # ------------------------ Definir Parametros ------------------
    FORMAT = pyaudio.paInt16  # Definir formato
    CHANNELS = 1  # Escojer un canal
    RATE = 44100  # Criterio de niquisht
    CHUNK = 1024  # Datos
    duracion = 4.1  # Duracion en segundos
    archivo = "grabacion.wav"  # Nombre del archivo

    # ------------------------ INICIAMOS "pyaudio" -----------------------
    audio = pyaudio.PyAudio()

    # ------------------------ Proceso de grabacion -----------------------
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("grabando...")
    frames = []

    for i in range(0, int(RATE/CHUNK*duracion)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("grabación terminada")

    # DETENEMOS GRABACIÓN

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # ------------------- Creamos y guardamos el archivo ----------------
    waveFile = wave.open(archivo, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    predict()


def predict():
    global Audio_encontrado
    # Crear los arreglos de entrada y de salida

    X = []
    # Seleccionar audio
    fs, data = wavfile.read('grabacion.wav')

    Audio = data/np.amax(data)

    # Tomamos la longitud de la señal
    L = len(Audio)

    n = np.arange(0, L)/fs

    # plt.plot(n, Audio)
    # plt.show()

    gk = fourier.fft(Audio)           # Calculamos la FFT de la señal de audio
    M = abs(gk)                        # Tomamos la Magnitud de la FFT
    # /np.amax(M)       # Tomamos la mitad de los datos (simetría de la transformada)
    M = (M[0:L//2])
    F = abs(fs*np.arange(0, L//2)/L)

    X.append(M)

    # plt.plot(F,M)
    # plt.xlabel('Frecuencia (Hz)', fontsize='14')
    # plt.ylabel('Amplitud FFT', fontsize='14')
    # plt.show()
    Posm = np.where(M == np.max(M))
    print((Posm[0]-(Posm[0]/1000)*(26))/4)
    X = np.array(X, dtype="uint8")

    # Tomar audio de acuerdo con la selección
    Audio_Seleccionado = X[0:1, 0:88200]

    y = Modelo_Importado.predict(Audio_Seleccionado).round()
    print(y, 'esta')
    y = y.tolist()
    Audio_encontrado = Etiquetas[y[0].index(1.0)-1]
    print(Audio_encontrado)
    ttk.Label(frm, text=Audio_encontrado).grid(column=2, row=2)



# ------------------ Estimacion del modelo --------------------------
if __name__ == "__main__":
    root = Tk()
    root.geometry("300x150")
    root.title("Notas RNA")
    # Fondo = PhotoImage(file="music.png")
    # Label(root, text="jefr", image=Fondo).place(x=0, y=0)
    frm = ttk.Frame(root, padding=20)
    frm.grid()
    ttk.Label(frm, text="Menu").grid(column=0, row=0)
    ttk.Label(frm, text="     ").grid(column=1, row=0)
    ttk.Button(frm, text="Grabar", command=record).grid(column=2, row=0)
    ttk.Label(frm, text="     ").grid(column=3, row=0)
    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=4, row=0)
    ttk.Label(frm, text=" ").grid(column=2, row=1)
    ttk.Label(frm, text="La Nota es:   ").grid(column=1, row=2)
    ttk.Label(frm, text=Audio_encontrado).grid(column=2, row=2)
    root.mainloop()
