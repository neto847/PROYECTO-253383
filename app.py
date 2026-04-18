import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Configuración inicial
st.title("Laboratorio de Probabilidad - UP Chiapas")

# Menú lateral para navegar entre bloques
opcion = st.sidebar.selectbox("Selecciona la Distribución", ["Inicio", "Normal", "Binomial", "Poisson"])

if opcion == "Normal":
    st.header("Distribución Normal")
    st.write("La distribución normal es una curva continua en forma de campana.")

    # --- ENTRADA DE DATOS (Sliders) ---
    col1, col2 = st.columns(2)
    with col1:
        mu = st.slider("Media (mu)", min_value=0.0, max_value=100.0, value=50.0)
    with col2:
        sigma = st.slider("Desviación Estándar (sigma)", min_value=0.1, max_value=20.0, value=5.0)

    # --- LÓGICA MATEMÁTICA ---
    # Creamos un rango de datos para el eje X (4 desviaciones estándar a los lados)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    # Calculamos la PDF (Función de Densidad de Probabilidad)
    y = norm.pdf(x, mu, sigma)

    # --- VISUALIZACIÓN ---
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue', label=f'Normal(mu={mu}, sigma={sigma})')
    ax.fill_between(x, y, alpha=0.2, color='blue') # Rellenar la curva
    ax.set_title("Gráfica de Densidad de Probabilidad")
    ax.set_xlabel("Valor de la Variable")
    ax.set_ylabel("Densidad")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

    # --- CÁLCULO DE PROBABILIDAD (Bonus de aprendizaje) ---
    st.subheader("Calculadora de Probabilidad")
    valor_x = st.number_input("Calcula P(X <= x). Ingresa el valor de x:", value=float(mu))
    prob = norm.cdf(valor_x, mu, sigma)
    st.info(f"La probabilidad de que X sea menor o igual a {valor_x} es: **{prob:.4f}**")
