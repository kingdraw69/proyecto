
# Importar la biblioteca Streamlit para crear la aplicación web
import streamlit as st

# Importar las bibliotecas tradicionales de numpy y pandas
import numpy as np
import pandas as pd

uploaded_files = st.file_uploader("Elige tus archivos", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    st.write(ModeloArbol.bin)
    st.write(ModeloBosqie.bin)
    st.write(ModeloNB.bin)
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

# Crear un contenedor para la introducción#INTRODICCION

"""Este es un ejemplo de cómo realizar una pagina Web usando streamlite
para desplegar un modelo de Machine Learning después de haber sido entrenado 


#Crear proyecto en Visual Studio Code

Cree una carpeta y descargue todos los assests o recursos que va usar en la aplicacion
imagenes, modelo.bin, archivos de datos, entre otros

./nombre_proyecto
                 /modelo.bin
                 /proyecto.py
                 /imagen.jpg
                 /datos.csv
                 
 Los datos pueden estar en remoto y llamarlo en python con pandas.               
 
#Bibliotecas requeridas para ejecutar el Deployment (Despiliegue a producción)

En este archivo no requieres tener instalado en python todas lasbibliotecas utilizadas
en el entrenamiento, solo las básicas para la página web

#Caso especial scikit-learn

Nota: antes de ejecutar cualquier código verifique con qué versión scikit-learn entrenó el modelo
y con qué versión va a desplegar el mismo para el usuario final.

Recomiento ir al enviroment (ambiente) donde lo entrenó. Si lo hizo con Google Colab cree un nuevo notebook
y ejecute el siguiente código:
!pip show scikit-learn

Y haga lo mismo en el enviroment de su máquina local, es decir abra el terminal del VSC, <en menú View/Terminal> o con CTRL ñ
y ejecute 
pip show scikit-learn

Una vez verificado que son la misma Versión, podrá ejecutar el desplieqgue, de lo contrario no podrá usar el modelo en producción.

Si requiere actualizar recomiendo actualice Google Colab y vuelva a entrenar, y descargue los modelos salvados con joblib.
!pip install --upgrade scikit-learn

Si requiere instalar la última versión (cuando no esté instalada en su pc)
# pip install -U scikit-learn


#Otras bliotecas básicas

Estas son algunas bibliotecas básicas, las instalas desde el Terminal- CTRL ñ en VSC (Visual Studio Code) 
pip install joblib
pip install matplotlib
pip install seaborn
pip freeze


# Crear un Repositorio en Github 

Antes de iniciar a escribir puede hacer la gestión de versiones con Github el proyecto, o siga los pasos para crear el
Repositorio básico.

Vaya a la opcion Create a new repository,
Digiete el nombre Repository name, por ejemplo churn
y haga un check a la casilla de Add a README file

En esta carpeta vas a copiar todos los archivos de la carpeta del proyecto


#Ejecutar el nombre_proyecto.py 


Cuando ingrese a Terminal de VSC, verifique que está ubicado en la carpeta del proyeco de lo contrario cambie
el directorio como se muestra en el ejemplo:

prompt> cd c:/Users/adiaz/Documents/churn/Churn.py

Si está ubicado en la carpeta del proyecto podrá ejecutar el servidor web de streamlit.
Nota: El RUN del VSC NO lanza el servidor web, si usan el RUN puede ir validando la sintaxis y el debbug pero no lanza el servidor.

Use el siguiente comando en cualquier shell o en Terminal de VSC para ejecutar y lanzar la página.

prompt>streamlit run nombre_proyecto.py

Tan pronto como ejecute el script como se muestra arriba, un servidor web Streamlit local se lanza y 
la aplicación se abrirá en una nueva pestaña en tu navegador web predeterminado. 

"""

# Librerias a importar 

import streamlit as st

#importar las bibliotecas tradicionales de numpy y pandas

import numpy as np
import pandas as pd

#importar las biliotecas graficas e imágenes

import matplotlib.pyplot as plt
import seaborn as sn

#importar libreria de paralelizacion de modelos
import joblib as jb


#Configurar la pagina (llamada Document)
#  Esta información aparece en la pestaña del navegador y el favíco
#y además lanzamos el Documento divido en dos parte la barra lateral y la pagina.

#El archivo cliente.ico debe descargalo de internet o usar en web una aplicación
#para convertir una imagen en formato favico.
#por ejemplo: https://imagen.online-convert.com/es/convertir-a-ico
#y descargalo en la carpeta del proyecto.

#El menú_items, lo podrá visualizar en los tres puntos verticales al lado derecho de la página...

st.set_page_config(
  page_title="Predicción de deserción de clientes",
  page_icon="cliente.ico",
  initial_sidebar_state='auto',
  menu_items={
        'Report a bug': 'http://www.unab.edu.co',
        'Get Help': "https://docs.streamlit.io/get-started/fundamentals/main-concepts",
        'About': "# Alfredo Díaz. Inteligencia Artificial *Ejemplo de clase* Deployment!"
    }
  )


# Recordar que cada ve que cambien un valor de un botón, slider o entrada en la página, está vuelve a ejecutar
#y puede tomar mucho tiempo recargar los modelos o los datos, por eso se usa un decorador "@" para poner en caché 
#lo modelos de machine learning,
# El decorador cache_resource para almacenar en caché funciones que devuelven recursos globales 
# (por ejemplo, conexiones de bases de datos, modelos de aprendizaje automático).
#Recomendacón: Se pone antes del comando o función, en este caso creamos una función que cargue el (los) modelo(s)
# a usar en la aplicación

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

#Crear contenedores.

#Puedo crear un container o una división de la página para separar o dividor la página en varias partes
#Podemos usar una variable para crear el contenedor así:
#contenedorintroduccion=st.container(border=True) y usar todos los métodos con ese contenedor por ejemplo
#contenedorintroduccion.st.subheader(..)
#o usarlo con with st.container() y llamar los diferente métodos dentro del with, recuerde dejar la tabulación.


with st.container( border=True):
  st.subheader("Modelo Machine Learning para predecir la deserción de clientes")
  #Se desea usar emoji lo puedes buscar aqui.
  st.write("Realizado por Alfredo Díaz Claros:wave:")
  st.write("""

**Introducción** 
cliente de nuestros sueños es el que permanece fiel a la empresa, comprando siempre sus productos o servicios. Sin embargo, en la realidad, 
los clientes a veces deciden alejarse de la empresa para probar o empezar a comprar otros productos o servicios y esto puede ocurrir en 
cualquier fase del customer journey. Sin embargo, existen varias medidas para prevenir o gestionar mejor esta circunstancia. Por eso lo mejor
es tener una herramienta predictiva que nos indique el estado futuro de dichos clientes usando inteligencia artificial, tomar las acciones 
de retenció necesaria. Constituye pues esta aplicación una herramienta importante para la gestión del marketing.

Los datos fueron tomados con la Información de la base de datos CRM de la empresa ubicada en Bucaramanfa,donde se
preparó 3 modelos de machine Learnig para predecir la deserció de clientes, tanto actuales como nuevos.

Datos Actualizados en la fuente: 20 de Marzo del 2024


Se utilizó modelos supervidados de clasificacion  tanto Naive Bayes, Arboles de decisión y Bosques Aleatorios 
entendiendo que hay otras técnicas, es el resultado de la aplicacion practico del curso de inteligencia artificial en estos modelos
revisado en clase. Aunqe la aplicación final sería un solo modelo, aqui se muestran los tres modelos para 
comparar los resultados.

 """)
  
  

#Vamos a crear otro contenedor y ese mismo contenedor por partimos en tres columnas
# con el comando st.columns(3), se la asignamos a 3 variables cada columna para llamarla

with st.container(border=True,height=300):
  st.subheader("Detalles")
  st.write(""" Este es un ejemplo de despliegue de los modelos de Machine Learning entrenados en
           Google Colab con las librerias de scikit-learn par Naive Bayes, Arbles de Decisión y Bosques Aleatorios.
           En este notebook podrás verificar el preprocesamiento del dataset y el entrenamiento y las pruebas
           y scores obtenidos.
           https://colab.research.google.com/drive/1bevxqlT_gQsZTrokc2LleIh40YTlQc2y?usp=sharing""")
  
  left_column, center_column, right_column = st.columns(3)
  with left_column:
    st.subheader("Introducción")
    st.image("introduccion.png",width=100)
    st.write(
      """ El objetivo de este trabajo acadeémico es construir una herramienta en código Python para predecir la deserción y requiere de las
      siguientes caracteristicas parra predecir:      
      'COMP', 'PROM', 'COMINT', 'COMPPRES', 'RATE', 'DIASSINQ', 'TASARET', 'NUMQ', 'RETRE'.
      El modelo elegido sería Naive Bayes, pero vamos a predecir en los tres modelos solo para efectos de comparación.
      
Dentro del universo de la analítica predictiva, existen numerosos modelos que, basados en Inteligencia Artificial, ayudan a las organizaciones a dar un paso más allá en su Data Journey y resolver sus problemas de negocio de manera efectiva.
Los modelos predictivos más conocidos (y potentes) son los de regresión y clasificación:

Los modelos de regresión nos permiten predecir un valor. Por ejemplo, cuál es el beneficio estimado que obtendremos de un determinado cliente (o segmento) en los próximos meses o nos ayudan a estimar el forecast de ventas.
Los modelos de clasificación en cambio nos permiten predecir la pertenencia a una clase.Por ejemplo, clasificar entre nuestros clientes quiénes son más propensos a una compra, a un abandono o a un fraude.
Y dentro de estos últimos, encontramos el modelo predictivo tipo churn: aquel que te ofrece información sobre qué clientes tienen más probabilidad de abandonarte. ¿Cómo funciona? Este modelo combina una serie de variables con datos históricos de tus clientes junto con datos de la situación actual. Los resultados son binarios: obtendremos un sí o un no (en forma de 0 y 1) en base a su grado de probabilidad de abandono.

Como todos los modelos predictivos, será indispensable ir reentrenándolo con nuevos datos conforme vaya pasando el tiempo para que no pierda fiabilidad y evitar que quede desactualizado.

Pese a que el modelo churn en sí mismo es valioso, en Keyrus trabajamos combinando diferentes casos de uso que nos ayuden a crear esa visión 360º del cliente tan buscada y deseada por todas las compañías, como podrían ser la propensión a la compra o el análisis del carrito de la compra, entre otros.

Este tipo de modelos para predecir la propensión al abandono te aportarán beneficios como:

Activar acciones de marketing más efectivas al conocer qué grupo de clientes es susceptible de dejar de comprarte.

Aumentar el CLTV de tus clientes, lo que se traduce en una reducción el CAC y una mayor rentabilidad al contar con esos clientes durante más tiempo.

Potenciar el branding de tu compañía al conseguir tener clientes más fieles, e incluso, transformarlos de manera natural en embajadores de tu marca.

Conocer más y mejor a tus clientes, lo que se traducirá en iterar la estrategia de cara a ser cada vez más customer-centric.

Tomar decisiones más estratégicas de cara a optimizar procesos y campañas. """)
    

  with center_column:
      st.subheader("Librerías genericas usadas")
      st.image("cliente.png",width=100)
      code = '''
      import pandas as pd 
      import numpy as np 
      mport csv
      import matplotlib.pyplot as plt
      import seaborn as sns  
      import joblib as jb
     '''
      st.code(code, language="python", line_numbers=True)
      

  with right_column:
      st.subheader("Librerías ML Scikit Learn usadas")
      st.image("modulos.png",width=100)
      code = '''
      
      #Dividir el dataset
      from sklearn.model_selection import train_test_split
      
      #importo las métricas
      
      from sklearn.metrics import confusion_matrix
      from sklearn.metrics import ConfusionMatrixDisplay
      from sklearn.metrics import classification_report
      from sklearn.metrics import roc_curve,auc
      
      #Librerias de validacion cruzada
      from sklearn.model_selection import KFold
      from sklearn.model_selection import cross_val_score
      from sklearn.model_selection import cross_val_predict
      
      #Naive Bayes
      from sklearn.naive_bayes import GaussianNB
      
      #Arboles
      from sklearn import tree
      from sklearn.tree import plot_tree

      #bosque
      from sklearn.ensemble import RandomForestClassifier

     '''
      st.code(code, language="python", line_numbers=True)


# Vamos a crear las opciones para seleccionar el tipo de modelo

modeloA=['Naive Bayes', 'Arbol de Decisión', 'Bosque Aleatorio']
#Aqui creo un diccionario para indicar cuando el modelo devuleva un número , mostrar el equivalente
# a si se retira o no.

churn = {1 : 'Cliente se retirará', 0 : 'Cliente No se Retirará' }

#Empiezo a dar formato al sidebar o barra lateral


#Usando markdown puedo cambiarl el formato a los titulos, textos, imagenes y demás.

styleimagen ="<style>[data-testid=stSidebar] [data-testid=stImage]{text-align: center;display: block;margin-left: auto;margin-right: auto;width: 100%;}</style>"
st.sidebar.markdown(styleimagen, unsafe_allow_html=True)

st.sidebar.image("churn.JPG", width=300)

#este script es para centrar pero si no lo desea no necesita hacerlo
styletexto = "<style>h2 {text-align: center;}</style>"
st.sidebar.markdown(styletexto, unsafe_allow_html=True)
st.sidebar.header('Seleccione los datos de entrada')

#Vammos a crear una función para mostrar todas las variables laterales para ingresar los datos en el model entrenado
#QAqui vamos a usar varias opciones. Le pasamos por parámetro a la funcion el modelo.

def seleccionar(modeloL):

    #Opción para seleccionar el modelo en un combo box o opción desplegable

  st.sidebar.subheader('Selector de Modelo')
  modeloS=st.sidebar.selectbox("Modelo",modeloL)

  #Filtrar por COMP con un slider
  st.sidebar.subheader('Seleccione la COMP')
  COMPS=st.sidebar.slider("Seleccion",4000,18000,8000,100)
  
  #Filtrar por PROM
  st.sidebar.subheader('Selector del PROM')
  PROMS=st.sidebar.slider("Seleccion",   0.7, 9.0,5.0,.5)
  
  #Filtrar por COMINT
  st.sidebar.subheader('Selector de COMINT')
  COMINTS=st.sidebar.slider("Seleccione",1500,58000,12000,100)
  
  #Filtrar por COMPPRES
  st.sidebar.subheader('Selector de COMPPRES') 
  COMPPRESS=st.sidebar.slider('Seleccione', 17000,90000,25000,100)
  
  #Filtrar por RATE
  st.sidebar.subheader('Selector de RATE')
  RATES=st.sidebar.slider("Seleccione",0.5,4.2,2.0,0.1)

  #Filtrar por DIASSINQ
  st.sidebar.subheader('Selector de DIASSINQ')
  DIASSINQS=st.sidebar.slider("Seleccione", 270,1800,500,10)
  
    #Filtrar por TASARET
  st.sidebar.subheader('Selector de TASARET')
  TASARETS=st.sidebar.slider("Seleccione",0.3,1.9,0.8,.5)
  
    #Filtrar por NUMQ
  st.sidebar.subheader('Selector de NUMQ')
  NUMQS=st.sidebar.slider("Seleccione",3.0,10.0,4.0,0.5)
  
    #Filtrar por departamento
  st.sidebar.subheader('Selector de RETRE entre 3 y 30')
  #RETRES=st.sidebar.slider("Seleccione",3.3,35.0,20.0,.5)
  RETRES=st.sidebar.number_input("Ingrese el valor de RETRE", value=3.3, placeholder="Digite el numero...")
  
  return modeloS,COMPS, PROMS, COMINTS ,COMPPRESS, RATES, DIASSINQS,TASARETS, NUMQS, RETRES


# Se llama la función, y se guardan los valores seleccionados en cada variable

modelo,COMP, PROM, COMINT ,COMPPRES, RATE, DIASSINQ,TASARET, NUMQ, RETRE=seleccionar(modeloA)

#Creamos un container para mostrar los resultados de predicción en el modelo que seleccione

with st.container(border=True):
  st.subheader("Predición")
  st.title("Predicción de Churn")
  st.write(""" El siguiente es el pronóstico de la deserción usanDo el modelo
           """)
  st.write(modelo)
  st.write("Se han seleccionado los siguientes parámetros:")
  # Presento los parámetros seleccionados en el slidder
  lista=[[COMP, PROM, COMINT ,COMPPRES, RATE, DIASSINQ,TASARET, NUMQ, RETRE]]
  X_predecir=pd.DataFrame(lista,columns=['COMP', 'PROM', 'COMINT', 'COMPPRES', 'RATE', 'DIASSINQ','TASARET', 'NUMQ', 'RETRE'])
  st.dataframe(X_predecir)
  
  #Creo un condicional para ejecutar el predict con base en el modelo seleccionado
  #Aqui estamos usando predict_proba para mostrar la probalidad de cada clase.
  # y tambien importancia para si el usiario quiere ver la importancia de cada variable en la predicción, en el modelo.
  
  
  if modelo=='Naive Bayes':
      y_predict=modeloNB.predict(X_predecir)
      probabilidad=modeloNB.predict_proba(X_predecir)
      importancia=pd.DataFrame()
  elif modelo=='Arbol de Decisión':
      y_predict=modeloArbol.predict(X_predecir)
      probabilidad=modeloArbol.predict_proba(X_predecir)
      importancia=modeloArbol.feature_importances_
      features=modeloArbol.feature_names_in_
  else :
      y_predict=modeloBosque.predict(X_predecir)
      probabilidad=modeloBosque.predict_proba(X_predecir)
      importancia=modeloBosque.feature_importances_
      features=modeloBosque.feature_names_in_
    
    
  #Cambiarmos el formato de header del container
  # recordar que classes_ muestra que el modelo devolverá dos clases la 0 para decir que no deserta
  # y la 1 para indicar que deserta
  
  styleprediccion= '<p style="font-family:sans-serif; color:Green; font-size: 42px;">La predicción es</p>'
  st.markdown(styleprediccion, unsafe_allow_html=True)
  prediccion='Resultado: '+ str(y_predict[0])+ "    - en conclusion :"+churn[y_predict[0]]
  st.header(prediccion+'   :warning:')
  
  st.write("Con la siguiente probabilidad")
  
  #Creamos dos columnas para mostrar las probabilidades de la predcción
  # la variable probabilidad es una matriz de dos columnas asi el valor
  # probabilidad[0][0] se refiere a la fila 0, y la columna 0, es decir el primer valor
  # probabilidad[0][1] se refiere a la fila 0, y la columna 1, es decir el segundo valor
 
  
  col1, col2= st.columns(2)
  col1.metric(label="Probalidad de NO :", value="{0:.2%}".format(probabilidad[0][0]),delta=" ")
  col2.metric(label="Probalidad de SI:", value="{0:.2%}".format(probabilidad[0][1]),delta=" ")
  
  st.write("La importancia de cada Factor en el modelo es:")
  if modelo!='Naive Bayes':
    importancia=pd.Series(importancia,index=features)
    st.bar_chart(importancia)  
  else:
    st.write("Naive Bayes no tiene parámetro de importancia de los features")


#Una vez terminado el código y de probarlo localmente, copie todos los archivo en su reporsitorio de github.
#Debe crear una cuenta en Streamlit.
#Una vez creada la cuenta debes vincularla a una cuenta de Github. 
