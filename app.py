import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm #esto pera el normal
from scipy.stats import binom # este para el binomial
from scipy.stats import poisson #esto actualiza al blque de poisson
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


    # este bloque es de lo binomial

elif opcion == "Binomial":
    st.header("Distribución Binomial")
    st.write("Se usa para experimentos con solo dos resultados posibles (éxito/fracaso).")

    # --- ENTRADA DE DATOS ---
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Número de ensayos (n)", min_value=1, max_value=100, value=10)
    with col2:
        p = st.slider("Probabilidad de éxito (p)", min_value=0.0, max_value=1.0, value=0.5)

    # --- LÓGICA MATEMÁTICA ---
    # En discretas, usamos valores enteros de 0 a n
    x = np.arange(0, n + 1)
    # Usamos PMF (Probability Mass Function) porque es discreta
    y = binom.pmf(x, n, p)

    # --- VISUALIZACIÓN ---
    fig, ax = plt.subplots()
    # Usamos bar() en lugar de plot() para resaltar que es discreta
    bars = ax.bar(x, y, color='orange', alpha=0.7, edgecolor='black')
    ax.set_title(f"Distribución Binomial (n={n}, p={p})")
    ax.set_xlabel("Número de éxitos")
    ax.set_ylabel("Probabilidad")
    ax.set_xticks(x) # Que se vean todos los números en el eje X
    
    st.pyplot(fig)

    # --- CÁLCULO DE PROBABILIDAD ---
    st.subheader("Calculadora de Probabilidad")
    exitos = st.number_input("¿Cuál es la probabilidad de tener exactamente 'k' éxitos?", 
                             min_value=0, max_value=int(n), value=int(n/2))
    
    prob_k = binom.pmf(exitos, n, p)
    st.success(f"La probabilidad de tener exactamente {exitos} éxitos es: **{prob_k:.4f}**")
   
   #este bloque es de poisson
elif opcion == "Poisson":
    st.header("Distribución de Poisson")
    st.write("Modela el número de eventos en un intervalo fijo de tiempo o espacio.")

    # --- ENTRADA DE DATOS ---
    mu_poisson = st.number_input("Promedio de eventos (Lambda - λ)", min_value=0.1, max_value=50.0, value=5.0)

    # --- LÓGICA MATEMÁTICA ---
    # Poisson puede ir hasta el infinito, pero graficamos hasta un punto razonable
    x = np.arange(0, int(mu_poisson * 3) + 1)
    y = poisson.pmf(x, mu_poisson)

    # --- VISUALIZACIÓN ---
    fig, ax = plt.subplots()
    ax.bar(x, y, color='green', alpha=0.7, edgecolor='black')
    ax.set_title(f"Distribución de Poisson (λ={mu_poisson})")
    ax.set_xlabel("Número de eventos")
    ax.set_ylabel("Probabilidad")
    
    st.pyplot(fig)

    # --- CÁLCULO DE PROBABILIDAD ---
    st.subheader("Calculadora de Probabilidad")
    k = st.number_input("¿Probabilidad de observar exactamente 'k' eventos?", 
                        min_value=0, value=int(mu_poisson))
    
    prob_k = poisson.pmf(k, mu_poisson)
    st.info(f"La probabilidad de observar exactamente {k} eventos es: **{prob_k:.4f}**")
