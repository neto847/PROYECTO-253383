import streamlit as st

st.title("proyecto de probabilidad -enresto")
st.write("hola aca estamos haciendp la prueba")

opcion = st.sidebar.selectbox("seleciona una etapa", ["INICIO", "DISTRIBUCION","HIPOTESIS"])
st.write(f"has seleccionado: {opcion}")


