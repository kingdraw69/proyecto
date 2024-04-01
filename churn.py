# Importar la biblioteca Streamlit para crear la aplicación web
import streamlit as st

# Importar las bibliotecas tradicionales de numpy y pandas
import numpy as np
import pandas as pd

# Importar las bibliotecas gráficas e imágenes
import matplotlib.pyplot as plt
import seaborn as sns

# Importar la biblioteca de paralelización de modelos
import joblib as jbjb

# Configurar la página
st.set_page_config(
  page_title="Predicción de deserción de clientes",
  page_icon="cliente.ico",
  initial_sidebar_state='auto',
  menu_items={
        'Report a bug': 'http://www.unab.edu.co',
        'Get Help': "https://docs.streamlit.io/get-started/fundamentals/main-concepts",
        'About': "# Ivan Eliecer Tarazona Rios. Inteligencia Artificial *Ejemplo de clase* Deployment!"
    }
  )
@st.cache_resource
def load_models():
  modeloNB=jb.load('modeloNB.bin')
  modeloArbol=jb.load('ModeloArbol.bin')
  modeloBosque=jb.load('ModeloBosque.bin')
  return modeloNB,modeloArbol,modeloBosque
#Cargamos a los modelos a usarse en el proyecto. 
#Nota, generalmente usanmos solo un modelo, pero para ejemplo académico lo vamos a hacer con 
#los tres modelo entrenado pero recuerde que se escoje el que mejore score nos ofrezca

modeloNB,modeloArbol,modeloBosque= load_models()
#Los siguientes textos aplican a nivel de la página.
st.title("Aplicación de predicción")
st.header('Machine Learning para Churn', divider='rainbow')
st.subheader('Ejemplo en los modelos :blue[Arbol de Decisión, Bosque Aleatorio y Naive Bayes]')

# Crear un contenedor para la introducción
with st.container(border=True):
    st.subheader("Introducción")
    st.write("""
        Este es un ejemplo de despliegue de modelos de Machine Learning entrenados en Google Colab con las librerías de scikit-learn para Naive Bayes, Árboles de Decisión y Bosques Aleatorios.
        En este notebook podrás verificar el preprocesamiento del dataset, el entrenamiento y las pruebas realizadas, así como los scores obtenidos.
        [Enlace al notebook](https://colab.research.google.com/drive/1bevxqlT_gQsZTrokc2LleIh40YTlQc2y?usp=sharing)

        **Introducción al proyecto**
        El cliente de nuestros sueños es aquel que permanece fiel a la empresa, comprando siempre sus productos o servicios. Sin embargo, en la realidad, los clientes a veces deciden alejarse de la empresa para probar o empezar a comprar otros productos o servicios, y esto puede ocurrir en cualquier fase del customer journey. Por eso, es importante tener una herramienta predictiva que nos indique el estado futuro de dichos clientes mediante inteligencia artificial, para tomar las acciones de retención necesarias. Esta aplicación constituye una herramienta importante para la gestión del marketing.

        Los datos fueron tomados de la base de datos CRM de una empresa ubicada en Bucaramanga. Se prepararon 3 modelos de machine learning para predecir la deserción de clientes, tanto actuales como nuevos.

        Datos Actualizados en la fuente: 20 de Marzo del 2024

        Se utilizaron modelos supervisados de clasificación, específicamente Naive Bayes, Árboles de Decisión y Bosques Aleatorios, entendiendo que existen otras técnicas. Este es el resultado de la aplicación práctica del curso de inteligencia artificial en estos modelos, revisado en clase. Aunque la aplicación final usaría solo un modelo, aquí mostramos los tres modelos para comparar los resultados.
    """)

# Crear un contenedor para mostrar detalles
with st.container(border=True, height=300):
    st.subheader("Detalles")
    st.write("""
        Este es un ejemplo de despliegue de los modelos de Machine Learning entrenados en Google Colab con las librerías de scikit-learn para Naive Bayes, Árboles de Decisión y Bosques Aleatorios.
        En este notebook podrás verificar el preprocesamiento del dataset, el entrenamiento y las pruebas realizadas, así como los scores obtenidos.
        [Enlace al notebook](https://colab.research.google.com/drive/1bevxqlT_gQsZTrokc2LleIh40YTlQc2y?usp=sharing)
    """)

    # Columnas para mostrar imágenes y códigos
    left_column, center_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Introducción")
        st.image("introduccion.png", width=100)
        st.write("""
            El objetivo de este trabajo académico es construir una herramienta en código Python para predecir la deserción, la cual requiere de las siguientes características para predecir:
            'COMP', 'PROM', 'COMINT', 'COMPPRES', 'RATE', 'DIASSINQ', 'TASARET', 'NUMQ', 'RETRE'.
            El modelo elegido será Naive Bayes, pero vamos a predecir con los tres modelos solo para efectos de comparación.

            Dentro del universo de la analítica predictiva, existen numerosos modelos basados en inteligencia artificial que ayudan a las organizaciones a resolver sus problemas de negocio de manera efectiva.
            Los modelos de regresión nos permiten predecir un valor, como el beneficio estimado que obtendremos de un determinado cliente (o segmento) en los próximos meses.
            Los modelos de clasificación nos permiten predecir la pertenencia a una clase, como clasificar entre nuestros clientes quiénes son más propensos a una compra, a un abandono o a un fraude.
            Y dentro de estos últimos, encontramos el modelo predictivo de churn, el cual nos ofrece información sobre qué clientes tienen más probabilidad de abandonarnos.
        """)

    with center_column:
        st.subheader("Librerías genéricas usadas")
        st.image("cliente.png", width=100)
        code = '''
        import pandas as pd
        import numpy as np
        import csv
        import matplotlib.pyplot as plt
        import seaborn as sns
        import joblib as jb
        '''
        st.code(code, language="python", line_numbers=True)

    with right_column:
        st.subheader("Librerías ML Scikit Learn usadas")
        st.image("modulos.png", width=100)
        code = '''
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        from sklearn.metrics import classification_report
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import KFold
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import cross_val_predict
        from sklearn.naive_bayes import GaussianNB
        from sklearn import tree
        from sklearn.tree import plot_tree
        from sklearn.ensemble import RandomForestClassifier
        '''
        st.code(code, language="python", line_numbers=True)

# Seleccionar el modelo y los parámetros
modeloA = ['Naive Bayes', 'Árbol de Decisión', 'Bosque Aleatorio']
churn = {1: 'Cliente se retirará', 0: 'Cliente No se Retirará'}

# Función para mostrar las opciones de selección de modelo
def seleccionar(modeloL):
    st.sidebar.subheader('Selector de Modelo')
    modeloS = st.sidebar.selectbox("Modelo", modeloL)

    st.sidebar.subheader('Seleccione la COMP')
    COMPS = st.sidebar.slider("Seleccion", 4000, 18000, 8000, 100)

    st.sidebar.subheader('Selector del PROM')
    PROMS = st.sidebar.slider("Seleccion", 0.7, 9.0, 5.0, .5)

    st.sidebar.subheader('Selector de COMINT')
    COMINTS = st.sidebar.slider("Seleccione", 1500, 58000, 12000, 100)

    st.sidebar.subheader('Selector de COMPPRES')
    COMPPRESS = st.sidebar.slider('Seleccione', 17000, 90000, 25000, 100)

    st.sidebar.subheader('Selector de RATE')
    RATES = st.sidebar.slider("Seleccione", 0.5, 4.2, 2.0, 0.1)

    st.sidebar.subheader('Selector de DIASSINQ')
    DIASSINQS = st.sidebar.slider("Seleccione", 270, 1800, 500, 10)

    st.sidebar.subheader('Selector de TASARET')
    TASARETS = st.sidebar.slider("Seleccione", 0.3, 1.9, 0.8, .5)

    st.sidebar.subheader('Selector de NUMQ')
    NUMQS = st.sidebar.slider("Seleccione", 3.0, 10.0, 4.0, 0.5)

    st.sidebar.subheader('Selector de RETRE entre 3 y 30')
    RETRES = st.sidebar.number_input("Ingrese el valor de RETRE", value=3.3, placeholder="Digite el numero...")

    return modeloS, COMPS, PROMS, COMINTS, COMPPRESS, RATES, DIASSINQS, TASARETS, NUMQS, RETRES

# Se llama la función, y se guardan los valores seleccionados en cada variable
modelo, COMP, PROM, COMINT, COMPPRES, RATE, DIASSINQ, TASARET, NUMQ, RETRE = seleccionar(modeloA)

# Crear un contenedor para mostrar los resultados de predicción
with st.container(border=True):
    st.subheader("Predicción")
    st.title("Predicción de Churn")
    st.write(f"El siguiente es el pronóstico de la deserción usando el modelo {modelo}")
    st.write("Se han seleccionado los siguientes parámetros:")
    lista = [[COMP, PROM, COMINT, COMPPRES, RATE, DIASSINQ, TASARET, NUMQ, RETRE]]
    X_predecir = pd.DataFrame(lista, columns=['COMP', 'PROM', 'COMINT', 'COMPPRES', 'RATE', 'DIASSINQ', 'TASARET', 'NUMQ', 'RETRE'])
    st.dataframe(X_predecir)

    # Crear un condicional para ejecutar el predict con base en el modelo seleccionado
    if modelo == 'Naive Bayes':
        y_predict = modeloNB.predict(X_predecir)
        probabilidad = modeloNB.predict_proba(X_predecir)
        importancia = pd.DataFrame()
    elif modelo == 'Árbol de Decisión':
        y_predict = modeloArbol.predict(X_predecir)
        probabilidad = modeloArbol.predict_proba(X_predecir)
        importancia = modeloArbol.feature_importances_
        features = modeloArbol.feature_names_in_
    else:
        y_predict = modeloBosque.predict(X_predecir)
        probabilidad = modeloBosque.predict_proba(X_predecir)
        importancia = modeloBosque.feature_importances_
        features = modeloBosque.feature_names_in_

    st.write("La predicción es:")
    prediccion = 'Resultado: ' + str(y_predict[0]) + "    - en conclusión: " + churn[y_predict[0]]
    st.header(prediccion)

    st.write("Con la siguiente probabilidad")
    col1, col2 = st.columns(2)
    col1.metric(label="Probabilidad de NO:", value="{0:.2%}".format(probabilidad[0][0]), delta=" ")
    col2.metric(label="Probabilidad de SI:", value="{0:.2%}".format(probabilidad[0][1]), delta=" ")

    st.write("La importancia de cada Factor en el modelo es:")
    if modelo != 'Naive Bayes':
        importancia = pd.Series(importancia, index=features)
        st.bar_chart(importancia)
    else:
        st.write("Naive Bayes no tiene parámetro de importancia de los features")
