import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import pyaudio
import wave
import scipy.fftpack as fourier
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import time
import tensorflow.keras as keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import load_model

# Crear RNA
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class RedNeuronal:
    def __init__(self, x, y):
        self.Entrada = x
        self.Objetivo = y

    def Entrenar(self):
        self.Modelo = Sequential()
        # Defina la cantidad de capas ocultas y las dimensiones de la entrada
        # En este caso es la longitud de la imagen vectorizada
        self.Modelo.add(Dense(100, activation='sigmoid',input_shape=(len(self.Entrada[1]),)))
        self.Modelo.add(Dense(512,activation='sigmoid'))
      


        
        # Configure la capa de salida
        self.Modelo.add(Dense(len(self.Objetivo[1]), activation='softmax'))
        # Configure los parámetros de entrenamiento del modelo
        self.Modelo.compile(loss="categorical_crossentropy",
                            optimizer="sgd",
                            metrics=['accuracy'])
        # Entrenar el modelo
        self.Modelo.fit(self.Entrada, self.Objetivo, batch_size=50, epochs=5000)
        # Guardar RNA en una carpeta
        self.Modelo.save('Mi_RNA')

    def Validar(self):
        # Importar RNA
        Modelo_Importado = load_model('Mi_RNA')
        # Evaluar el modelo
        loss, acc = Modelo_Importado.evaluate(self.Entrada, self.Objetivo)
        print('Precisión:', acc)
        print(self.Modelo.predict(self.Entrada).round())


if __name__ == "__main__":
    Etiquetas = ['DO', 'RE','MI', 'FA', 'SOL', 'LA', 'SI']  # Crear los arreglos de entrada y de salida
    y = [1,2,3,4,5,6,7]   # A cada etiqueta asigne un numero como identificador
    X = []
    # Leer imagenes desde una ubicación
    for i in Etiquetas:
        fs, data = wavfile.read('Audios/'+i+'.wav')  # Cargar los audios
        # Convertir imagen en una matriz de valores numéricos
        Audio_m = data/np.amax(data)

        L = len(Audio_m)                 # Tomamos la longitud de la señal
        n = np.arange(0, L)/fs

        # plt.plot(n, Audio_m)
        # plt.show()

        # Calculamos la FFt de la señal de audio
        gk = fourier.fft(Audio_m)
        M_gk = abs(gk)                   # Tomamos la Magnitud de la FFT
        # Tomamos la mitad de los datos (recordar la simetría de la transformada)
        M_gk = (M_gk[0:L//2])
        F = abs(fs*np.arange(0, L//2)/L)  # Calcular la frecuencia de la señal

        # plt.plot(F, M_gk)
        # plt.xlabel('Frecuencia (Hz)', fontsize='14')
        # plt.ylabel('Amplitud FFT', fontsize='14')
        # plt.show()
        X.append(M_gk)

    # Convertir los valores de la matriz en enteros
    X = np.array(X, dtype="uint8")
    X = X.astype("float32")/255.0    # Normalizar el valor para procesarlo
    # Vectorizar los valores para entrenar la red
    X = X.reshape(7, -1)
    # Convetir los valores de salida en variables categoricas
    y = np.array(y, dtype="uint8")
    y = to_categorical(y)            # Crear la red

    RNA = RedNeuronal(X, y)
    RNA.Entrenar()
    RNA.Validar()
