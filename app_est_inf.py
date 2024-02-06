## Librerias
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import geopandas as gpd
import numpy as np
import time
from datetime import datetime, timedelta, date
import math
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import oracledb
import sqlalchemy as sa
from sqlalchemy import select, and_, func
import statsmodels.api as sm
from patsy import dmatrices
import re
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN

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
        'About': "Modulo creado para el análisis de incidentes en SUANET"
    }
)
################################# CONSULTA ORACLE
@st.cache_data
def consult_data(fecha_inicial0, fecha_final0, dias):
    fecha_inicial = fecha_inicial0.strftime('%Y-%m-%d %H:%M:%S')
    fecha_final = fecha_final0.strftime('%Y-%m-%d %H:%M:%S')
    ## CONEXIÓN A LA BASE DE DATOS
    dialect = 'oracle'
    sql_driver = 'oracledb'
    ## ORACLE SDM ## hacer esto con variables de entorno
    un = 'BITACORA'
    host = "172.30.6.21"
    port = "1521"
    sn = "BITACORA"
    pw = 'B1tac0r2023*'
    # try:
    if (fecha_final0-fecha_inicial0).days <31:
        to_engine: str = dialect + '+' + sql_driver + '://' + un + ':' + pw + '@' + host + ':' + str(port) + '/?service_name=' + sn
        connection = sa.create_engine(to_engine)
        query = f"SELECT INCIDENTNUMBER, LATITUDE, LONGITUDE, INCIDENTDATE FROM MV_INCIDENT WHERE INCIDENTDATE BETWEEN TO_TIMESTAMP('{fecha_inicial}', 'YYYY-MM-DD HH24:MI:SS') AND TO_TIMESTAMP('{fecha_final}', 'YYYY-MM-DD HH24:MI:SS')"
        # test_df = pd.read_sql_query('SELECT INCIDENTNUMBER, LATITUDE, LONGITUDE, INCIDENTDATE FROM MV_INCIDENT WHERE INCIDENTDATE BETWEEN TO_TIMESTAMP("' + str(fecha_inicial) + '") AND TO_TIMESTAMP("'+ str(fecha_final)+'"), "YYYY-MM-DD HH24:MI:SS"', connection)
        test_df = pd.read_sql_query(query, connection)
        ## Selección especifica de dias
        # trans_days = []
        # for d in dias:
        #     if d == 'Lunes':
        #         trans_days.append('Monday')
        #     elif d == 'Martes':
        #         trans_days.append('Tuesday')
        #     elif d == 'Miercoles':
        #         trans_days.append('Wednesday')
        #     elif d == 'Jueves':
        #         trans_days.append('Thursday')
        #     elif d == 'Viernes':
        #         trans_days.append('Friday')
        #     elif d == 'Sabado':
        #         trans_days.append('Saturday')
        #     elif d == 'Domingo':
        #         trans_days.append('Sunday')
        # if 'Todos los dias' not in dias:
        #     test_df = test_df[test_df['incidentdate'].dt.day_name().isin(trans_days)]
        #test_df = pd.read_sql_query('SELECT INCIDENTNUMBER, LATITUDE, LONGITUDE, INCIDENTDATE FROM MV_INCIDENT', connection)
        test_df = test_df[test_df['latitude']!=0]
        test_df = gpd.GeoDataFrame(test_df, geometry=gpd.points_from_xy(test_df.longitude, test_df.latitude), crs="EPSG:4326")
        areas = gpd.read_file("data/shp_areas_pol.shp.zip")
        areas['geo_area'] = areas['geometry']
        test_df = test_df.sjoin(areas, how='left',predicate='within')
        st.write(str(test_df.shape[0]) + " incidentes consultados entre " + str(test_df.incidentdate.min()) + " y " + str(test_df.incidentdate.max()))
        return test_df
    else:
        st.error("Por favor seleccione máximo un mes de datos.")
        return pd.DataFrame()  
    # except:
    #     st.error("Sin datos recuperados. Por favor verificar fechas.")
    #     return pd.DataFrame()
        

################################# FUNCIONES AUXILIARES

def bn_model(df):
    df['MONTH'] = df['incidentdate'].dt.month
    df['WEEK_DAY'] = df['incidentdate'].dt.day_name()
    def p(x):
        if (x > 4) and (x <= 12):
            return 'Mañana'
        elif (x > 12) and (x <= 20 ):
            return 'Tarde'
        elif (x > 20) and (x <= 24):
            return'Noche'
        elif (x > 0) and (x <= 4) :
            return 'Noche'
    df['periodo_dia'] = df['incidentdate'].dt.hour.apply(p)
    df['area_policia'] = df['area_polic']
    df['Incidente'] = df['incidentnumber']
    dreg = df.groupby(['WEEK_DAY','area_policia','MONTH','periodo_dia']).agg({'Incidente':'count'}).reset_index()
    ## Regresion
    expr = """Incidente ~ C(periodo_dia) +C(WEEK_DAY, Treatment(reference='Sunday')) + C(MONTH) + area_policia +area_policia:periodo_dia + area_policia:WEEK_DAY"""
    y, x = dmatrices(expr, dreg, return_type='dataframe')
    nb2_training_results = sm.GLM(y, x,family=sm.families.NegativeBinomial(alpha=0.012165)).fit()
    res = pd.DataFrame(nb2_training_results.params[np.where(nb2_training_results.pvalues < 0.15)[0]]).reset_index()
    res.columns = ['parametros','betas']
    return res

def filter_params(x):
        x=x['parametros']
        if (re.search(r"C\(WEEK_DAY",x)) and not (re.search(":",x)):
            y = re.search(r"\[T\.(.*)\]",x).group(1)
            clase = "Dia de la semana"
        elif (re.search(r"area_policia\[",x)) and not (re.search(":",x)):
            y = re.search(r"\[T\.(.*)\]",x).group(1)
            clase = "Area policia"
        elif (re.search(r"periodo_dia",x)) and not (re.search(":",x)):
            y = re.search(r"\[T\.(.*)\]",x).group(1)
            clase = "Periodo del dia"
        elif (re.search(r"C\(WEEK_DAY",x)) and (re.search(":",x)):
            y = re.search(r"\[T\.(.*)\]:.*\[T\.(.*)\]",x).group(1) +" - "+ re.search(r"\[T\.(.*)\]:.*\[T\.(.*)\]",x).group(2)
            clase = "Combinaciones"
        elif (re.search(r"area_policia\[",x)) and (re.search(":",x)):
            y = re.search(r"\[(.*)\]:.*\[T\.(.*)\]",x).group(1) +" - "+  re.search(r"\[(.*)\]:.*\[T\.(.*)\]",x).group(2)
            clase = "Combinaciones"
        elif (re.search(r"periodo_dia",x)) and (re.search(":",x)):
            y = re.search(r"\[T\.(.*)\]:.*\[T\.(.*)\]",x).group(1) +" - "+  re.search(r"\[T\.(.*)\]:.*\[T\.(.*)\]",x).group(2)
            clase = "Combinaciones"
        else:
            y = x
            clase = "Sin match"
        return [y, clase]
################################# PAGINA DE INICIO    
##################################################

## Show in webpage
st.markdown(
    """
    # RECOMENDACIONES A PARTIR DE ANÁLISIS DESCRIPTIVOS E INFERENCIALES
    """
    )
col1, col2 = st.columns([1,1])
inicial_date = col1.date_input("Seleccione la fecha desde la cual quiere tomar información de incidentes: ", date.today()- timedelta(days=8)) #date(2022, 12, 1)
final_date = col2.date_input("Seleccione la fecha hasta la cual quiere tomar información de incidentes: ", date.today()- timedelta(days=1))#'today')
#dias = col1.multiselect("Selecciona los dias de la semana especificos entre las fechas para los análisis ",['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo','Todos los dias'], default='Todos los dias', help='Este campo filtra los dias entre las fehcas de inicio y fin seleccionadas arriba. Se pueden seleccionar una o varias',placeholder='Seleccione una o varias opciones')
dias = 0
def streamlit_menu():
    selected = option_menu(
                menu_title=None,  # required
                options=["Centroides", "Inferencia dias", "Inferencia áreas","Otras inferencias", "Clustering"],  # required
                menu_icon="cast",  # optional
                default_index=0,  # optional
                orientation="horizontal",
            )
    return selected

selected = streamlit_menu()

if selected == "Centroides":
    st.title(f"Centroides por área de la policia de los datos consultados")
    data = consult_data(inicial_date, final_date, dias)
    if not data.empty:
        data = data[data['latitude']!=0]
        centroides_areas=data.dissolve(by='area_polic').centroid
        areas = data.groupby('area_polic').agg({'geo_area':'first','incidentnumber':'count'}).reset_index()
        # Centroide de todos los incidentes por areas de la policia
        fig = px.scatter_mapbox(centroides_areas, width=1000, height=800,
                                lat=centroides_areas.geometry.y, lon=centroides_areas.geometry.x,
                                #hover_name="area_polic",
                                labels="area_policia",
                                center={"lat": 4.62, "lon": -74.15},
                                mapbox_style="carto-positron",
                                color_continuous_scale=px.colors.cyclical.IceFire,
                                zoom=10,
        )

        fig2 = px.choropleth_mapbox(areas, width=1000, height=800,
                                    geojson=areas.geo_area,
                                    locations=areas.index,
                                    color=areas.incidentnumber,
                                    center={"lat": 4.62, "lon": -74.15},
                                    mapbox_style="carto-positron",
                                    color_continuous_scale=px.colors.cyclical.IceFire,
                                    zoom=10,
                                    hover_name=areas.area_polic,
                                    opacity=0.3,
        )
        fig.add_trace(fig2.data[0])
        fig.update_layout(colorscale_diverging='twilight')#colorscale=dict())#coloraxis=dict(cmax=6, cmin=3))
        #st.write(areas)
        st.plotly_chart(fig, use_container_width=True)
    
if selected == "Inferencia dias":
    st.title(f"Modelo inferencial - Recomendaciones por dia")
    data = consult_data(inicial_date, final_date, dias)
    if not data.empty:
        res = bn_model(data)
        res['clasif'] = ''
        res[['parametros','clasif']] = res.apply(filter_params,axis=1, result_type="expand")
        res1 = res[res['clasif']=='Dia de la semana']
        res2 = res[res['clasif']=='Periodo del dia']
        fig1 = px.bar(res1, x='parametros', y='betas', width=400, height=400)
        fig2 = px.bar(res2, x='parametros', y='betas', width=400, height=400)
        cola, col1, col2, colb = st.columns([1,2,2,1])
        col1.markdown('### Análisis por días (Respecto al domingo)')
        col1.plotly_chart(fig1, use_container_width=True)
        col2.markdown('### Análisis por parte del día (Respecto a la mañana)')
        col2.plotly_chart(fig2, use_container_width=True)

# if selected == "Inferencia meses":
#     st.title(f"Modelo inferencial - Recomendaciones por meses")
#     data = consult_data(inicial_date, final_date)
#     if not data.empty:
#         res = bn_model(data)
#         st.write(res)
#         res['clasif'] = ''
#         res[['parametros','clasif']] = res.apply(filter_params,axis=1, result_type="expand")
#         res = res[res['clasif']=='Area policia']
#         fig1 = px.bar(res, x='parametros', y='betas', width=400, height=400)
#         cola, col1, colb = st.columns([1,2,1])
#         col1.markdown('### Análisis por días')
#         col1.plotly_chart(fig1, use_container_width=True)
if selected == "Inferencia áreas":
    st.title(f"Modelo inferencial - Recomendaciones por áreas de la policia")
    data = consult_data(inicial_date, final_date, dias)
    if not data.empty:
        res = bn_model(data)
        res['clasif'] = ''
        res[['parametros','clasif']] = res.apply(filter_params,axis=1, result_type="expand")
        res = res[res['clasif']=='Area policia']
        fig1 = px.bar(res, x='parametros', y='betas', width=400, height=400)
        cola, col1, colb = st.columns([1,2,1])
        col1.markdown('### Análisis por áreas de la policia (Respecto al AREA 1)')
        col1.plotly_chart(fig1, use_container_width=True)
if selected == "Otras inferencias":
    st.title(f"Modelo inferencial - Otras inferencias")
    data = consult_data(inicial_date, final_date, dias)
    if not data.empty:
        res = bn_model(data)
        res['clasif'] = ''
        res[['parametros','clasif']] = res.apply(filter_params,axis=1, result_type="expand")
        res = res[res['clasif']=='Combinaciones']
        res.sort_values('betas', inplace=True)
        
        cola, col1, colb = st.columns([1,2,1])
        fig = go.Figure(data=go.Heatmap(z=res['betas'], x=res['betas'], y=res['parametros']))
        fig.update_layout( autosize=False, width=800, height=800, )
        col1.markdown("## Otras inferencias cruzadas singnificativas")
        col1.plotly_chart(fig)
if selected == "Clustering":
    st.title(f"Modelo de Clustering")
    data = consult_data(inicial_date, final_date, dias)
    if not data.empty:
        X = np.array(data[['longitude', 'latitude']], dtype='float64')
        model = DBSCAN(eps=0.001, min_samples=25).fit(X)
        class_predictions = model.labels_
        data['CLUSTERS_DBSCAN'] = class_predictions
        data1 = data[data['CLUSTERS_DBSCAN']>1]
        fig = px.scatter_mapbox(data1, width=1000, height=1000,
                                lat=data1.geometry.y, lon=data1.geometry.x,
                                color = 'CLUSTERS_DBSCAN',
                                #hover_name="area_polic",
                                labels="area_policia",
                                center={"lat": 4.62, "lon": -74.15},
                                mapbox_style="carto-positron",
                                color_continuous_scale=px.colors.cyclical.IceFire,
                                zoom=11,
        )
        fig2 = px.density_mapbox(data1, lat='latitude', lon='longitude', z='CLUSTERS_DBSCAN', radius=4,width=1000, height=1000,
                        center=dict(lat=4.62, lon=-74.15), zoom=11,
                        mapbox_style="carto-positron")
        fig.add_trace(fig2.data[0])
        st.plotly_chart(fig,  use_container_width=True)
        st.write(f'Number of clusters found: {len(np.unique(class_predictions))}')
        st.write(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')
# with st.expander("Ver explicación"):
#     st.markdown(
#     """
#     Para la estadistica inferencial se plantearon modelos de regresión que permitieran conocer la relevancia estadistica de las variables explicativas:
    
#     Se establece define un modelo de regresión binomial negativa. Al suponer que la cantidad de incidentes sigue una distribución binomial negativa de tipo:
    
#     $$ CantIncidentes_{i} \\approx BN(r,p) $$
#     """
#     )
#     st.latex(r'''
#         P(CantIncidentes)_{i}=y_{i} =  \begin{pmatrix}
#             y_{i} +r-1 \\
#             r-1
#             \end{pmatrix}p^r(1-p)^y_{i}
#         ''')
#     st.markdown("""Bajo este supuesto, se plantea el siguiente modelo:""")
#     st.latex(r'''
#         log(\lambda) = \beta _{0} + \beta _{1}Dia + \beta_ {2}Mes + \beta_{3}Area_{pol} + \beta_{4}Periodo_{dia} + \beta_{5}Area_{pol}Periodo_{dia} + \beta_{6}Area_{pol}Dia
#         ''')
    
#     ## Imagen modelo
#     image = Image.open('img/pred_regresion.jpg')
#     #Imagen centrada
#     col1, col2, col3 = st.columns([1,6,1])
#     col1.write("")
#     col2.image(image, caption='Predicciones regresion')
#     col3.write("")
    
    
    
#     st.markdown("""Del anterior modelo se extrae la información evidenciada abajo.""")
# st.markdown(
#     """    
#     ## 1. Centroides: Puntos centrales a los historicos de incidentes en cada area
#     """)

# col1, col2 = st.columns([3, 8])

# with col2:
#     # INSERTANDO IMAGEN HTML DE CENTROIDES
#     path_to_html = "img/centroides.html" 
#     with open(path_to_html,'r', encoding='latin-1') as f: 
#         html_data = f.read()
        
#     #Imagen centrada
#     cola, colb, colc = st.columns([1,3,1])
#     cola.write("")
#     with colb:
#         st.components.v1.html(html_data,width=900, height=800)
#     colc.write("")
#     #st.components.v1.html(html_data,width=900, height=800)

# with col1:
#     for i in range(20):
#         st.text("  ")
#     st.markdown(
#     """
#     Se pueden observar los centroides de incidentes en cada area de la policia cerca a los cuales se recomienda ubicar las unidades para atender de manera mas efciente los eventos.
#     """
#     )
    
# st.markdown(
#     """
#     ## 3. Resultados inferidos del modelo
    
#     1. Inferencias temporales:
#         - En la tarde se generan 10% más incidentes respecto a la mañana
#         - En la noche se generan 27% menos incidentes respecto a la mañana
#     """)
# image = Image.open('img/inferencial_por_mes.jpg')
# #Imagen centrada
# col1, col2, col3 = st.columns([1,5,1])
# col1.write("")
# col2.image(image, caption='inferencial por mes')
# col3.write("")

# image = Image.open('img/inferencial_semana.jpg')
# #Imagen centrada
# col1, col2, col3 = st.columns([1,5,1])
# col1.write("")
# col2.image(image, caption='inferencial_semana')
# col3.write("")

# st.markdown(
#     """
#     2. Inferencias espaciales:
#     """)
# image = Image.open('img/inf_espaciales.jpg')
# #Imagen centrada
# col1, col2, col3 = st.columns([1,6,1])
# col1.write("")
# col2.image(image, caption='inf_espaciales')
# col3.write("")



# st.markdown(
#     """
#     ## 2. Resultados inferidos del modelo
#     """
# )
# col3, col4 = st.columns([3, 1])
# with col3:
#     # INSERTANDO IMAGEN HTML DE CENTROIDES
#     image = Image.open('img/est_inferencial.jpg')
#     #Imagen centrada
#     col1, col2, col3 = st.columns([1,6,1])
#     col1.write("")
#     col2.image(image, caption='Tabla de inferenciales')
#     col3.write("")

# with col4:
#     for i in range(10):
#         st.text("  ")
#     st.markdown(
#     """
#     Se observa para cada area algunos de los resultados de las estadisticas inferenciales resultado del modelo de regresión binomial negativo a partir de los datos de las bitacoras.
#     """
#     )




