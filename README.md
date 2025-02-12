# Lab-2-PDS
Informe de laboratorio número 2 Procesamiento Digital de Señales

# Laboratorio de Convolución, Correlación y Transformada de Fourier

Este repositorio contiene el código y análisis de herramientas fundamentales del procesamiento digital de señales: **Convolución**, **Correlación** y **Transformada de Fourier**. El laboratorio incluyó cálculos manuales y simulaciones en Python para una mejor comprensión de estos conceptos.

---

## **Librerias**

```python
!pip install wfdb
import wfdb
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch, savgol_filter
````
a) Convolución
La convolución es utilizada para determinar la respuesta de un sistema lineal e invariante en el tiempo ante una señal de entrada. En este caso, se tomó un sistema ℎ[𝑛] y una señal 𝑥[𝑛] para calcular la señal de salida 𝑦[𝑛] mediante la convolución.

El procedimiento fue el siguiente:

Cálculo manual: Se determinó la señal resultante 𝑦[𝑛] usando sumatorias, evaluando el comportamiento de la salida en función de los valores de entrada y del sistema.

Implementación en Python: Se utilizó la función numpy.convolve() para obtener 𝑦[𝑛] y se imprimieron los valores resultantes.

Visualización de la señal resultante: Se obtuvo una representación gráfica utilizando Matplotlib. El gráfico permite observar el comportamiento secuencial de 𝑦[𝑛].

-Maria Paula:

Usando sumatorias:

<img src="https://github.com/user-attachments/assets/c38323e4-0bff-4845-8e2a-b643841fcee3" alt="Texto alternativo" width="600">

```phyton
h = [5, 6, 0, 0, 9, 1, 2]  # Sistema h[n]
x = [1, 0, 7, 2, 6, 4, 2, 3, 1, 1]  # Señal x[n]
y = np.convolve(x, h)
print("\nResultado de la convolución:\ny[n]:", y)
#Grafica
plt.figure(figsize=(8, 5))
plt.stem(np.arange(len(y)), y, linefmt="r-", markerfmt="ro", basefmt=" ")
plt.title("\nResultado de la convolución $y[n]$", fontsize=14)
plt.xlabel("$n$", fontsize=12)
plt.ylabel("$y[n]$", fontsize=12)
plt.grid(True)
plt.show()
````
<img src="https://github.com/user-attachments/assets/1d158d73-d760-4e3c-a558-5da0d6338f22" alt="Texto alternativo" width="600">

-Juan Pablo:

Usando sumatorias:

<img src="https://github.com/user-attachments/assets/bc923ff0-830c-47d9-9c0d-32e6053d6817" alt="Texto alternativo" width="600">

```phyton
h = [5, 6, 0, 0, 7, 9, 5]  # Sistema h[n]
x = [1, 0, 7, 6, 7, 3, 8, 4, 9, 5]  # Señal x[n]
y = np.convolve(x, h)
print("\nResultado de la convolución:\ny[n]:", y)
#Grafica
plt.figure(figsize=(8, 5))
plt.stem(np.arange(len(y)), y, linefmt="g-", markerfmt="go", basefmt=" ")
plt.title("\nResultado de la convolución $y[n]$", fontsize=14)
plt.xlabel("$n$", fontsize=12)
plt.ylabel("$y[n]$", fontsize=12)
plt.grid(True)
plt.show()
```
<img src="https://github.com/user-attachments/assets/6c150b78-a640-489a-b5e8-603d67999f26" alt="Texto alternativo" width="600">


-Paula Vanessa

Usando sumatorias:

<img src="https://github.com/user-attachments/assets/77c385ab-fd68-4928-bd72-469b7b595c45" alt="Texto alternativo" width="600">


```phyton
h = [5, 6, 0, 0, 8, 4, 4]  # Sistema h[n]
x = [1, 0, 3, 1, 6, 4, 3, 0, 2, 4]  # Señal x[n]
y = np.convolve(x, h)
print("\nResultado de la convolución:\ny[n]:", y)
#Grafica
plt.figure(figsize=(8, 5))
plt.stem(np.arange(len(y)), y, linefmt="b-", markerfmt="bo", basefmt=" ")
plt.title("\nResultado de la convolución $y[n]$", fontsize=14)
plt.xlabel("$n$", fontsize=12)
plt.ylabel("$y[n]$", fontsize=12)
plt.grid(True)
plt.show()
````
<img src="https://github.com/user-attachments/assets/6441f668-6eb9-4f64-a42e-9902bcde6099" alt="Texto alternativo" width="600">


b) Correlación:

La correlación es utilizada para medir la similitud entre dos señales en diferentes desplazamientos temporales (lags). En este caso, se analizaron dos señales definidas por funciones trigonométricas: una señal coseno y una señal seno, con una frecuencia de 100 Hz y un período de muestreo de 1.25 ms.

El procedimiento para calcular la correlación cruzada de estas señales fue:

Definición de parámetros: Se estableció el período de muestreo y el rango de índices de muestreo, que determinan los puntos en los que se evalúan las señales.

Generación de las señales: Se calcularon los valores de 𝑥1[𝑛] como una función coseno y 𝑥2[𝑛]como una función seno para los mismos instantes de tiempo discretos.

Cálculo de la correlación cruzada: Se aplicó la función numpy.correlate() para determinar la relación entre ambas señales a medida que una de ellas se desplaza con respecto a la otra. Esto permite analizar cómo una señal influye en la otra en distintos instantes de tiempo.

Por ultimo se generaron gráficos que muestran las dos señales originales, lo que permitio observar las formas y comportamiento en el dominio y la correlación, que indica en qué desplazamientos hay mayor similitud entre las señales

```phyton
# 1. Definir parámetros
Ts = 1.25e-3  # Periodo de muestreo
n = np.arange(0, 9)  # Rango de índices de muestreo (0 ≤ n < 9)
t = n * Ts  # Vector de tiempo discreto

# 2. Definir las señales x1[nTs] y x2[nTs]
x1 = np.cos(2 * np.pi * 100 * t)  # Señal x1[nTs]
x2 = np.sin(2 * np.pi * 100 * t)  # Señal x2[nTs]

# 3. Calcular la correlación cruzada
correlacion = np.correlate(x1, x2, mode="full")  # Correlación cruzada
lags = np.arange(-len(x1) + 1, len(x1))  # Definir los desplazamientos (lags)

# 4. Mostrar resultados
print("x1[n]:", x1)
print("x2[n]:", x2)
print("Correlación cruzada r_x1x2[n]:", correlacion)

# 5. Graficar señales y correlación
plt.figure(figsize=(8, 8))

# Gráfico de x1[n]
plt.subplot(3, 1, 1)
plt.stem(n, x1, linefmt="r-", markerfmt="ro", basefmt=" ")
plt.title(r"Señal $x_1[n] = \cos(2\pi 100 n T_s)$", fontsize=14, color='r')
plt.xlabel("$n$", fontsize=12)
plt.ylabel("$x_1[n]$", fontsize=12)
plt.grid()

# Gráfico de x2[n]
plt.subplot(3, 1, 2)
plt.stem(n, x2, linefmt="g-", markerfmt="go", basefmt=" ")
plt.title(r"Señal $x_2[n] = \sin(2\pi 100 n T_s)$", fontsize=14, color='g')
plt.xlabel("$n$", fontsize=12)
plt.ylabel("$x_2[n]$", fontsize=12)
plt.grid()

# Gráfico de la correlación cruzada
plt.subplot(3, 1, 3)
plt.stem(lags, correlacion, linefmt="b-", markerfmt="bo", basefmt=" ")
plt.title(r"Correlación cruzada $r_{x_1x_2}[n]$", fontsize=14, color='b')
plt.xlabel("Desplazamiento (lag)", fontsize=12)
plt.ylabel("$r_{x_1x_2}[n]$", fontsize=12)
plt.grid()

plt.tight_layout()
plt.show()
````
Resultado de la correlación:

x1[n]: [ 1.00000000e+00  7.07106781e-01  6.12323400e-17 -7.07106781e-01
 -1.00000000e+00 -7.07106781e-01 -1.83697020e-16  7.07106781e-01
  1.00000000e+00]
x2[n]: [ 0.00000000e+00  7.07106781e-01  1.00000000e+00  7.07106781e-01
  1.22464680e-16 -7.07106781e-01 -1.00000000e+00 -7.07106781e-01
 -2.44929360e-16]
Correlación cruzada r_x1x2[n]: [-2.44929360e-16 -7.07106781e-01 -1.50000000e+00 -1.41421356e+00
 -2.54671001e-16  2.12132034e+00  3.50000000e+00  2.82842712e+00
 -2.28847549e-17 -2.82842712e+00 -3.50000000e+00 -2.12132034e+00
  4.55531587e-16  1.41421356e+00  1.50000000e+00  7.07106781e-01
  0.00000000e+00]

![image](https://github.com/user-attachments/assets/fc1796be-af17-4d53-97fd-5943fc542cc0)

c) Caracterización de la señal en el tiempo:

Para caracterizar la señal en el dominio del tiempo, se siguieron los siguientes pasos:

Carga de la señal: Se utilizó la función wfdb.rdrecord() para leer la señal descargada de PhysioNet.

Obtención de la frecuencia de muestreo: Se obtuvo la frecuencia de muestreo 𝑓𝑠, que define cuántas muestras por segundo tiene la señal.

Selección de la ventana de análisis: Se definió una duración de observación de 5 segundos. Se calcularon las muestras necesarias para ese intervalo en función de la frecuencia de muestreo.

Diezmado de la señal: Se aplicó un factor de diezmado para facilitar la visualización y reducir la cantidad de datos. Se seleccionaron muestras cada cierto intervalo, ajustando el tiempo de acuerdo con el nuevo muestreo efectivo.
```phyton
record = wfdb.rdrecord("S0088_ST_V2")
senal = record.p_signal[:, 0]  # Tomar la primera señal
fs = record.fs  # Frecuencia de muestreo

duracion_mostrar = 5  # segundos
muestras_mostrar = int(duracion_mostrar * fs)
factor_diezmado = 8
senal_mostrar = senal[:muestras_mostrar:factor_diezmado]
tiempo_mostrar = np.arange(len(senal_mostrar)) * (factor_diezmado/fs)
```
Visualización de la señal

Se generó una gráfica utilizando funciones de Python que muestra la evolución de la señal en función del tiempo.
```phyton
plt.figure(facecolor='white')
plt.plot(tiempo_mostrar, senal_mostrar, color='midnightblue', linewidth=1.5)
plt.xlabel("Tiempo (s)", fontsize=12)
plt.ylabel("Amplitud", fontsize=12)
plt.title("Señal de EMG", fontsize=14, pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(-1.0, 1.0)
plt.xlim(0, 5)
plt.margins(x=0.02)
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/9df2d240-1662-4c04-a1ed-d60d11c71cd6)


Estadísticos descriptivos

Para analizar la señal en el dominio del tiempo, se utilizaron funciones de Python para calcular:

Media (np.mean()): Valor promedio de la señal.
Mediana (np.median()): Valor central de la señal.
Desviación estándar (np.std()): Medida de la dispersión respecto a la media.
Coeficiente de variación: Relación entre la desviación estándar y la media.
```phyton
# Cálculo de estadísticos descriptivos en función del tiempo
media = np.mean(senal)  # Frecuencia media
mediana = np.median(senal)  # Frecuencia mediana
desviacion = np.std(senal)  # Desviación estándar
coef_variacion = desviacion / media  # Coeficiente de variación

print(f"\nEstadísticas de la señal:")
print(f"Media: {media:.6f}")  # Frecuencia media solicitada
print(f"Mediana: {mediana:.6f}")  # Frecuencia mediana solicitada
print(f"Desviación estándar: {desviacion:.6f}")  # Desviación estándar solicitada
print(f"Coeficiente de variación: {coef_variacion:.6f}")
````
Estadísticas de la señal:
Media: -0.001089
Mediana: -0.001614
Desviación estándar: 0.101831
Coeficiente de variación: -93.532295
Descripción de la señal EMG según su clasificación

La señal electromiográfica (EMG) es una señal eléctrica que refleja la actividad muscular, generada por la activación de las fibras musculares a través de impulsos neuronales. En este caso, los datos fueron obtenidos de PhysioNet, una base de datos biomédica que generalmente contiene registros de EMG intramuscular.

La señal EMG no es periódica ni sinusoidal, sino una señal estocástica, caracterizada por picos y fluctuaciones rápidas debido a la activación asincrónica de múltiples unidades motoras dentro del músculo.

Su amplitud típica varía entre 0.1 mV y 10 mV, dependiendo del músculo y la intensidad de la contracción.

Para capturar la señal EMG, la frecuencia de muestreo suele estar en el rango de 500 Hz a 10 kHz.

La Transformada de Fourier (FFT) permite obtener el espectro de frecuencias, revelando la distribución de energía dentro del rango característico de la EMG intramuscular.

A partir del espectro de potencia, se pueden calcular parámetros estadísticos clave, como la frecuencia media, mediana y desviación estándar, los cuales ofrecen información sobre la actividad muscular y el estado de fatiga del músculo.

Transformada de Fourier y densidad espectral

Se utilizaron funciones de Python para transformar la señal al dominio de la frecuencia mediante:

Transformada de Fourier (np.fft.fft()): Permite obtener las componentes espectrales de la señal.
Frecuencias asociadas (np.fft.fftfreq()): Determina la escala de frecuencias correspondientes a la transformada.
Densidad espectral de potencia (welch() de SciPy): Proporciona una estimación de la distribución de energía en cada frecuencia.
Se generaron dos gráficas:

Transformada de Fourier, que muestra la magnitud de las frecuencias presentes.
Densidad espectral de potencia, que representa la distribución de energía en el espectro.
```phyton
# Aplicar la Transformada de Fourier
frecuencias = np.fft.fftfreq(len(senal_mostrar), d=1/fs * factor_diezmado)
transformada = np.fft.fft(senal_mostrar)

# Obtener la densidad espectral de potencia
frecuencias_welch, densidad_espectral = welch(senal_mostrar, fs=fs/factor_diezmado, nperseg=256)

# Graficar la transformada de Fourier
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.plot(frecuencias[:len(frecuencias)//2], np.abs(transformada[:len(transformada)//2]), color='navy')
plt.title("Transformada de Fourier de la Señal")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()

# Graficar la densidad espectral de potencia
plt.subplot(1, 2, 2)
plt.semilogy(frecuencias_welch, densidad_espectral, color='navy')
plt.title("Densidad Espectral de Potencia")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de Potencia")
plt.grid()

plt.tight_layout()
plt.show()
````

![image](https://github.com/user-attachments/assets/e988aec8-9a2f-40b1-80bf-84cb99023c41)


Estadísticos descriptivos en función de la frecuencia

Se calcularon estadísticos descriptivos en funcion de la frecuencia, basándose en su transformada.

Utilizando Python, se realizaron los siguientes pasos:

Transformada de Fourier (np.fft.fft()): Se obtuvo el espectro de frecuencias de la señal.
Filtrado de frecuencias positivas: Se eliminaron las frecuencias negativas para un análisis más representativo.
Cálculo de estadísticos:

Frecuencia media: Se calculó ponderando las magnitudes espectrales.
Frecuencia mediana: Punto donde la energía acumulada de la señal se divide en dos partes iguales.
Desviación estándar de la frecuencia: Medida de dispersión de las frecuencias respecto a la media.
Tambien se generó un histograma de frecuencias, que muestra la distribución de la energía espectral a lo largo del espectro de frecuencias.

```phyton
# Aplicar la Transformada de Fourier
frecuencias = np.fft.fftfreq(len(senal_mostrar), d=1/fs * factor_diezmado)
transformada = np.fft.fft(senal_mostrar)
magnitudes = np.abs(transformada)  # Magnitud del espectro

# Filtrar solo frecuencias positivas
mask = frecuencias > 0
frecuencias_positivas = frecuencias[mask]
magnitudes_positivas = magnitudes[mask]

# Cálculo de estadísticos en función de la frecuencia
frecuencia_media = np.sum(frecuencias_positivas * magnitudes_positivas) / np.sum(magnitudes_positivas)
frecuencia_mediana = frecuencias_positivas[np.argsort(np.cumsum(magnitudes_positivas))[
    np.searchsorted(np.cumsum(magnitudes_positivas), np.sum(magnitudes_positivas) / 2)]]
desviacion_frecuencia = np.sqrt(np.sum(((frecuencias_positivas - frecuencia_media) ** 2) * magnitudes_positivas) / np.sum(magnitudes_positivas))

# Imprimir resultados
print("\nEstadísticos descriptivos en función de la frecuencia:")
print(f"Frecuencia media: {frecuencia_media:.6f} Hz")
print(f"Frecuencia mediana: {frecuencia_mediana:.6f} Hz")
print(f"Desviación estándar de la frecuencia: {desviacion_frecuencia:.6f} Hz")

# Graficar histograma de frecuencias
plt.figure(figsize=(6, 4))
plt.hist(frecuencias_positivas, weights=magnitudes_positivas, bins=30, alpha=0.7, color='slateblue', edgecolor='black')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud espectral")
plt.title("Histograma de frecuencias")
plt.grid()
plt.show()
````

Estadísticos descriptivos en función de la frecuencia:
Frecuencia media: 49.373557 Hz
Frecuencia mediana: 50.573028 Hz
Desviación estándar de la frecuencia: 25.244547 Hz



![image](https://github.com/user-attachments/assets/8a7f7817-7339-45d4-adac-236ecbb01f3a)

Conclusiones:

La práctica permitió aplicar y comprender tres herramientas fundamentales del procesamiento digital de señales:

-Convolución:

Se entendió como una operación entre una señal de entrada y un sistema para producir una señal de salida. Esta operación se realizó tanto manualmente como usando Python, facilitando la visualización del resultado en forma gráfica y secuencial. Este concepto es esencial para modelar cómo un sistema responde a estímulos externos.

-Correlación:

Se utilizó para medir la similitud entre dos señales, demostrando su utilidad en la detección de patrones y el análisis de relaciones temporales entre señales. La correlación se calculó para señales periódicas definidas por funciones seno y coseno, mostrando claramente cómo varía la relación entre ambas señales a medida que cambia el desfase.

-Transformada de Fourier:

A través del análisis espectral de una señal descargada, se exploró su comportamiento en el dominio de la frecuencia. Esto incluyó la generación de gráficos de la transformada de Fourier y su densidad espectral de potencia, así como el cálculo de estadísticos descriptivos como la frecuencia media, mediana y la desviación estándar. Este enfoque es clave para identificar las componentes de frecuencia predominantes en señales biomédicas y de otros tipos. En resumen, la práctica permitió no solo aprender a usar herramientas computacionales como Python para realizar estos cálculos, sino también comprender el significado práctico de estas operaciones en la interpretación de señales y sistemas. La integración de métodos manuales y computacionales refuerza la capacidad de análisis crítico y técnico en el campo del procesamiento digital de señales.

Referencias

Correlación cruzada y convolución. (n.d.). https://support.ptc.com/help/mathcad/r10.0/es/index.html#page/PTC_Mathcad_Help/convolution_and_cross_correlation.html

Martínez, M. (2021, May 12). Transformada de Fourier: qué es y cómo se calcula. Nobbot. https://www.nobbot.com/que-es-la-transformada-de-fourier-y-para-que-sirve/

C, L. a. L., & Toloza, D. C. (2020). Análisis frecuencial y de la densidad espectral de potencia de la estabilidad de sujetos amputados. TecnoLógicas, 23(48), 1–16. https://doi.org/10.22430/22565337.1453


