# Lab-2-PDS
Informe de laboratorio número 2 Procesamiento Digital de Señales

# Laboratorio de Convolución, Correlación y Transformada de Fourier

Este repositorio contiene el código y análisis de herramientas fundamentales del procesamiento digital de señales: **Convolución**, **Correlación** y **Transformada de Fourier**. El laboratorio incluyó cálculos manuales y simulaciones en Python para una mejor comprensión de estos conceptos.

---

## **Código**

A continuación, se presenta el código implementado para calcular la convolución, correlación y transformada de Fourier:

```python
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Cálculo de la Convolución
# -------------------------------

# Señales para convolución
h = [5, 6, 0, 0, 1, 4, 6]  # Sistema
x = [1, 0, 2, 1, 4, 5, 4, 8, 1, 9]  # Señal de entrada

# Convolución usando numpy
y = np.convolve(x, h)
print("Resultado de la convolución:", y)

# Graficar la convolución
plt.stem(y, use_line_collection=True)
plt.title("Resultado de la Convolución")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.show()

# -------------------------------
# 2. Cálculo de la Correlación
# -------------------------------

# Señales para correlación
Ts = 1.25e-3  # Período de muestreo
n = np.arange(0, 9)  # Valores de n
x1 = np.cos(2 * np.pi * 100 * n * Ts)  # Señal x1
x2 = np.sin(2 * np.pi * 100 * n * Ts)  # Señal x2

# Correlación cruzada
correlacion = np.correlate(x1, x2, mode='full')
print("Resultado de la correlación:", correlacion)

# Graficar la correlación
plt.plot(np.arange(-len(x1) + 1, len(x2)), correlacion)
plt.title("Resultado de la Correlación Cruzada")
plt.xlabel("Desplazamiento")
plt.ylabel("Correlación")
plt.grid()
plt.show()

# -------------------------------
# 3. Transformada de Fourier
# -------------------------------

# Simular una señal (descargada de PhysioNet o generada)
t = np.linspace(0, 1, 500)  # Tiempo
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # Señal combinada

# Transformada de Fourier
frecuencia = np.fft.fft(signal)
frecuencia_abs = np.abs(frecuencia)  # Magnitud de la transformada

# Graficar señal original
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title("Señal en el Dominio del Tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")

# Graficar transformada
frecuencias = np.fft.fftfreq(len(signal), d=(t[1] - t[0]))
plt.subplot(2, 1, 2)
plt.plot(frecuencias[:len(signal)//2], frecuencia_abs[:len(signal)//2])  # Solo parte positiva
plt.title("Transformada de Fourier")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()


## **Conclusión**

1. **Convolución**:  
   Se utilizó como operación para combinar una señal y un sistema, calculando la salida tanto manualmente como en Python. Esto permitió visualizar la respuesta del sistema de forma gráfica.

2. **Correlación**:  
   Se usó para analizar la similitud entre dos señales periódicas definidas con funciones seno y coseno, mostrando cómo varía la relación al cambiar el desfase.

3. **Transformada de Fourier**:  
   Mediante el análisis espectral de una señal descargada, se descompuso en sus componentes frecuenciales, generando gráficos como la densidad espectral de potencia y cálculos de estadísticos descriptivos.

---
