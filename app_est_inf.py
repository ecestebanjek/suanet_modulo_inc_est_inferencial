## Librerias
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

CURRENT_THEME = "light"
IS_DARK_THEME = False

################################ SET PAGE
st.set_page_config(
    page_title="SUANET - Modulo estadistica inferencial",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


################################# PAGINA DE INICIO    
##################################################

## Show in webpage
st.markdown(
    """
    # RECOMENDACIONES A PARTIR DE ANÁLISIS DESCRIPTIVOS E INFERENCIALES
    """
    )
with st.expander("Ver explicación"):
    st.markdown(
    """
    Para la estadistica inferencial se plantearon modelos de regresión que permitieran conocer la relevancia estadistica de las variables explicativas:
    
    Se establece define un modelo de regresión binomial negativa. Al suponer que la cantidad de incidentes sigue una distribución binomial negativa de tipo:
    
    $$ CantIncidentes_{i} \\approx BN(r,p) $$
    """
    )
    st.latex(r'''
        P(CantIncidentes)_{i}=y_{i} =  \begin{pmatrix}
            y_{i} +r-1 \\
            r-1
            \end{pmatrix}p^r(1-p)^y_{i}
        ''')
    st.markdown("""Bajo este supuesto, se plantea el siguiente modelo:""")
    st.latex(r'''
        log(\lambda) = \beta _{0} + \beta _{1}Dia + \beta_ {2}Mes + \beta_{3}Area_{pol} + \beta_{4}Periodo_{dia} + \beta_{5}Area_{pol}Periodo_{dia} + \beta_{6}Area_{pol}Dia
        ''')
    
    ## Imagen modelo
    image = Image.open('img/pred_regresion.jpg')
    #Imagen centrada
    col1, col2, col3 = st.columns([1,6,1])
    col1.write("")
    col2.image(image, caption='Predicciones regresion')
    col3.write("")
    
    
    
    st.markdown("""Del anterior modelo se extrae la información evidenciada abajo.""")
st.markdown(
    """    
    ## 1. Centroides: Puntos centrales a los historicos de incidentes en cada area
    """)

col1, col2 = st.columns([3, 8])

with col2:
    # INSERTANDO IMAGEN HTML DE CENTROIDES
    path_to_html = "img/centroides.html" 
    with open(path_to_html,'r', encoding='latin-1') as f: 
        html_data = f.read()
        
    #Imagen centrada
    cola, colb, colc = st.columns([1,3,1])
    cola.write("")
    with colb:
        st.components.v1.html(html_data,width=900, height=800)
    colc.write("")
    #st.components.v1.html(html_data,width=900, height=800)

with col1:
    for i in range(20):
        st.text("  ")
    st.markdown(
    """
    Se pueden observar los centroides de incidentes en cada area de la policia cerca a los cuales se recomienda ubicar las unidades para atender de manera mas efciente los eventos.
    """
    )
    
st.markdown(
    """
    ## 3. Resultados inferidos del modelo
    
    1. Inferencias temporales:
        - En la tarde se generan 10% más incidentes respecto a la mañana
        - En la noche se generan 27% menos incidentes respecto a la mañana
    """)
image = Image.open('img/inferencial_por_mes.jpg')
#Imagen centrada
col1, col2, col3 = st.columns([1,5,1])
col1.write("")
col2.image(image, caption='inferencial por mes')
col3.write("")

image = Image.open('img/inferencial_semana.jpg')
#Imagen centrada
col1, col2, col3 = st.columns([1,5,1])
col1.write("")
col2.image(image, caption='inferencial_semana')
col3.write("")

st.markdown(
    """
    2. Inferencias espaciales:
    """)
image = Image.open('img/inf_espaciales.jpg')
#Imagen centrada
col1, col2, col3 = st.columns([1,6,1])
col1.write("")
col2.image(image, caption='inf_espaciales')
col3.write("")



st.markdown(
    """
    ## 2. Resultados inferidos del modelo
    """
)
col3, col4 = st.columns([3, 1])
with col3:
    # INSERTANDO IMAGEN HTML DE CENTROIDES
    image = Image.open('img/est_inferencial.jpg')
    #Imagen centrada
    col1, col2, col3 = st.columns([1,6,1])
    col1.write("")
    col2.image(image, caption='Tabla de inferenciales')
    col3.write("")

with col4:
    for i in range(10):
        st.text("  ")
    st.markdown(
    """
    Se observa para cada area algunos de los resultados de las estadisticas inferenciales resultado del modelo de regresión binomial negativo a partir de los datos de las bitacoras.
    """
    )




