import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
from scipy.stats import norm #esto pera el normal
from scipy.stats import binom # este para el binomial
from scipy.stats import poisson #esto actualiza al blque de poisson
from scipy import stats #esto para las puebras de HIPOTESIS
import google.generativeai as genai #para la apy key de importacion de google
import pandas as pd
# Configuración de la IA
def configurar_ia(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

st.title("ERNESTO Laboratorio de Probabilidad - UP Chiapas")
# Esto va en el sidebar, arriba del selectbox de opciones
with st.sidebar:
    st.title(" Configuración de IA")
    user_api_key = st.text_input("Ingresa tu Google API Key", type="password")
# Menú lateral para navegar entre bloques
opcion = st.sidebar.selectbox("Selecciona  la etapa", ["Inicio", "Normal", "Binomial", "Poisson", "Hipotesis","Asistente IA","Cargar Datos"])
if opcion == "Inicio":
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR3ENXQR-FJ6rtFMXHNKbAXH2rQMRuA12ta9Q&s", width=200)
    st.title("Sistema de Análisis Estadístico")
    st.markdown("""
    Bienvenido al proyecto final de **Probabilidad y Estadística**. 
    Esta aplicación interactiva permite:
    * **Visualizar** distribuciones de probabilidad (Normal, Binomial, Poisson).
    * **Ejecutar** pruebas de hipótesis con veredictos en tiempo real.
    * **Consultar** a una Inteligencia Artificial para interpretar resultados.
    
    *Desarrollado por: Ernesto Diaz*
    """)

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


elif opcion == "Hipotesis":
    st.header(" Pruebas de Hipótesis para la Media")
    st.write("Calcula si hay evidencia suficiente para rechazar una afirmación sobre el promedio.")

    # --- 1. ENTRADA DE DATOS DEL PROBLEMA ---
    with st.expander(" Configuración del Experimento", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            mu_h0 = st.number_input("Media a probar (H0)", value=50.0)
            x_barra = st.number_input("Media observada (Muestra)", value=52.0)
        with col2:
            n_muestra = st.number_input("Tamaño de muestra (n)", min_value=2, value=30)
            sigma_s = st.number_input("Desviación estándar (s o σ)", value=5.0)
        
        alpha = st.select_slider("Nivel de significancia (α)", 
                                 options=[0.01, 0.05, 0.10], value=0.05)

    # --- 2. LÓGICA DE DECISIÓN (Z o T) ---
    # Calculamos el Error Estándar
    error_estandar = sigma_s / np.sqrt(n_muestra)
    
    # Calculamos el estadístico observado (Z o T se calculan igual)
    estadistico = (x_barra - mu_h0) / error_estandar

    if n_muestra > 30:
        tipo_prueba = "Z"
        # Cálculo de P-valor para dos colas (bilateral)
        p_valor = 2 * (1 - stats.norm.cdf(abs(estadistico)))
    else:
        tipo_prueba = "T"
        # Grados de libertad para la T de Student
        gl = n_muestra - 1
        p_valor = 2 * (1 - stats.t.cdf(abs(estadistico), df=gl))

    # --- 3. EL VEREDICTO ---
    st.subheader(f"Resultados de la prueba {tipo_prueba}")
    
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Estadístico observado", f"{estadistico:.4f}")
    col_res2.metric("P-Valor", f"{p_valor:.4f}")

    if p_valor < alpha:
        st.error(f" Como {p_valor:.4f} < {alpha}, **Rechazamos H0**.")
        st.write("Hay evidencia suficiente para decir que la media es diferente.")
    else:
        st.warning(f" Como {p_valor:.4f} >= {alpha}, **No rechazamos H0**.")
        st.write("No hay evidencia suficiente para cambiar la afirmación inicial.")
        # --- 4. GRÁFICA DE LA PRUEBA ---
    st.subheader(" Visualización de la Región de Rechazo")
    
    # Creamos datos para la curva normal estándar (Z)
    x_plot = np.linspace(-4, 4, 1000)
    y_plot = stats.norm.pdf(x_plot, 0, 1)

    fig2, ax2 = plt.subplots()
    ax2.plot(x_plot, y_plot, color='black', label='Distribución Normal Estándar')

    # Encontrar el valor crítico para la gráfica (bilateral)
    z_critico = stats.norm.ppf(1 - alpha/2)

    # Sombrear zonas de rechazo
    x_rechazo_der = np.linspace(z_critico, 4, 100)
    ax2.fill_between(x_rechazo_der, stats.norm.pdf(x_rechazo_der), color='red', alpha=0.5, label='Zona de Rechazo')
    
    x_rechazo_izq = np.linspace(-4, -z_critico, 100)
    ax2.fill_between(x_rechazo_izq, stats.norm.pdf(x_rechazo_izq), color='red', alpha=0.5)

    # Dibujar línea del estadístico observado
    ax2.axvline(estadistico, color='blue', linestyle='--', linewidth=2, label=f'Tu Estadístico: {estadistico:.2f}')
    
    ax2.legend()
    ax2.set_title("Prueba Bilateral (Dos Colas)")
    st.pyplot(fig2)

    # --- 5. INTERVALOS DE CONFIANZA (Mencionado en tu PDF del 14 de abril) ---
    st.divider()
    st.subheader(" Intervalo de Confianza")
    margen_error = z_critico * error_estandar
    li = x_barra - margen_error
    ls = x_barra + margen_error
    
    st.write(f"Con un **{100*(1-alpha):.0f}%** de confianza, la verdadera media está entre:")
    st.info(f" **({li:.4f}, {ls:.4f})**")
elif opcion == "Asistente IA":
    st.header("Asistente Estadístico Inteligente")
    
    if not user_api_key:
        st.warning("Por favor, ingresa tu API Key en la barra lateral.")
    else:
        try:
            # Configurar el modelo
            genai.configure(api_key=user_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            pregunta = st.text_area("Escribe tu duda para el profesor:", 
                                     placeholder="Ej. ¿Cómo interpreto un p-valor de 0.03?")
            
            if st.button("Consultar a Gemini"):
                if pregunta:
                    with st.spinner("El profesor está analizando tu duda..."):
                        # Enviamos la consulta
                        contexto = "Eres un profesor de estadística experto. Responde de forma clara, educativa y en español."
                        prompt_final = f"{contexto}\n\nPregunta del alumno: {pregunta}"
                        
                        response = model.generate_content(prompt_final)
                        
                        # Guardamos y mostramos el resultado inmediatamente
                        if response.text:
                            st.markdown("### 🎓 Respuesta del Profesor:")
                            st.info(response.text)
                        else:
                            st.error("Gemini no pudo generar una respuesta. Revisa tu conexión.")
                else:
                    st.error("Debes escribir una pregunta primero.")
                    
        except Exception as e:
            if "overloaded" in str(e).lower():
                st.warning("Los servidores de Google están saturados. Por favor, espera un momento y presiona el botón de nuevo.")
            else:
                st.error(f"Error: {e}")
elif opcion == "Cargar Datos":
    st.header("Análisis de Datos Reales (CSV)")
    st.write("Sube tu archivo para calcular automáticamente la media y desviación.")
    
    archivo_subido = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if archivo_subido is not None:
        df = pd.read_csv(archivo_subido)
        st.success("¡Archivo cargado con éxito!")
        
        # Mostrar una vista previa
        with st.expander("Ver tabla de datos"):
            st.dataframe(df)
        
        # Selección de columna
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if columnas_numericas:
            columna = st.selectbox("Selecciona la columna para analizar", columnas_numericas)
            
            # Cálculos automáticos
            x_barra = df[columna].mean()
            sigma_s = df[columna].std()
            n_muestra = len(df[columna])
            
            # Mostrar resultados
            col1, col2, col3 = st.columns(3)
            col1.metric("Media (x̄)", f"{x_barra:.2f}")
            col2.metric("Desviación (s)", f"{sigma_s:.2f}")
            col3.metric("Muestra (n)", n_muestra)
            
            st.info(" Ahora puedes usar estos valores en la sección de 'Hipótesis'.")
        else:
            st.error("El archivo no contiene columnas numéricas.")

    