import gensim
import pandas as pd
import regex as re
import nltk
import numpy as np
import logging # verbosidad (útil para saber qué está ocurriendo al ejecutar código)
from nltk.chat.util import Chat, reflections
import sklearn.ensemble # clasificador
import sklearn.metrics # evaluar desempeño de clasificador
import sklearn.model_selection # partición de conjuntos de entrenamiento - evaluación
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('spanish')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def cargar_datos_df(archivo):
    # Carga los archivos en un dataframe
    return pd.read_csv(archivo, sep="\t", header=None)

def cargar_datos(archivo):
    # Carga los archivos en un arreglo
    oraciones = []
    with open(ruta, encoding='utf-8', errors='ignore') as file:  # Usamos utf-8 para preservar los caracteres especiales
        for line in file:
            oraciones.append((line.rstrip()))  # Usamos la función strip para remover el caracter \n (nueva línea)
    return oraciones


def separar(archivo):
    # Separa preguntas de respuestas
    aux_preg = []
    aux_resp = []
    orden_preg = []
    orden_resp =[]
    data = pd.DataFrame({'orden_preg':orden_preg, 'preg':aux_preg,'ord_resp':orden_resp,'resp':aux_resp})
    for i in range(0, len(archivo)):
        m = i
        if m % 2:
            aux_resp.append(archivo.get(i))
            orden_resp.append(i)
        else:
            aux_preg.append(archivo.get(i))
            orden_preg.append(i)

    data['orden_preg'] = orden_preg
    data['preg'] = aux_preg
    data['ord_resp'] = orden_resp
    data['resp'] = aux_resp

    return data



def limpiar_datos(hola, nombre):
    hola[nombre] = hola[nombre].str.replace(u"á", "a")
    hola[nombre] = hola[nombre].str.replace(u"é", "e")
    hola[nombre] = hola[nombre].str.replace(u"í", "i")
    hola[nombre] = hola[nombre].str.replace(u"ó", "o")
    hola[nombre] = hola[nombre].str.replace(u"ú", "u")
    return hola



def vectorizer(text, model):
    # Vectorizamos
    vectors = []
    for i in text:
        try:
            vectors.append(model.wv[i])
        except:
            pass
    return(np.mean(vectors,axis=0))


def tokeniza(sentencias):
    # Tokenizamos las palabras
    arreglo = []
    arreglo = [nltk.word_tokenize(sentencia) for sentencia in sentencias]
    return arreglo

def remover_stopWords(arreglo):
    # Remueve las stopwords
    aux = []
    for palabra in arreglo:
        aux_stopwords_palabra = [word for word in palabra if word not in stopwords]
        aux.append(aux_stopwords_palabra)
    return aux




# Cargamos las coversaciones del archivo de texto
ruta = "archivo_chico.txt"
sentences = cargar_datos(ruta)


conversaciones = { i : sentences[i] for i in range(0, len(sentences) ) }

datos = separar(conversaciones)


datos['preg'] = datos['preg'].astype('str')
datos['resp'] = datos['resp'].astype('str')


datos['preg'] = datos['preg'].str.lower()
datos['resp'] = datos['resp'].str.lower()

datos= limpiar_datos(datos, 'preg')
datos= limpiar_datos(datos, 'resp')

print(datos.head(1))









