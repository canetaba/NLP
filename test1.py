import pandas as pd
import regex as re
import nltk
import numpy as np
import logging # verbosidad (útil para saber qué está ocurriendo al ejecutar código

nltk.download('stopwords')
nltk.download('punkt')

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
    for i in range(0, len(archivo)):
        m = i
        if m % 2:
            aux_resp.append(archivo.get(i))
        else:
            aux_preg.append(archivo.get(i))
    return(aux_preg,aux_resp)



def limpiar_datos(text):
    # Limpia el texto quitando tildes y reemplazando mayusculas por minusculas
    text = text.lower()  # Todo a minúsculas
    text = re.sub(r'[^A-Za-zñáéíóú]', ' ', text)  # Reemplaza todas los signos de puntuación por un espacio
    text = re.sub('á', 'a', text)  # Reemplaza las vocales con tilde por su forma basal
    text = re.sub('é', 'e', text)
    text = re.sub('í', 'i', text)
    text = re.sub('ó', 'o', text)
    text = re.sub('ú', 'u', text)
    return text


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
    stopwords = ['de', 'para', 'el', 'ella','la',
                 'los', 'las', 'es', 'lo','o','a',
                 'y','ya','le','que', 'una', 'un',
                 'con','se','por','mi','del','te','me','al','tu',
                 'en','su','pero', 'entre', 'sin', 'embargo',
                 'ni','les','nos','mas','ey','aj','oh','es']
    # Remueve las stopwords
    aux = []
    for palabra in arreglo:
        aux_stopwords_palabra = [word for word in palabra if word not in stopwords]
        aux.append(aux_stopwords_palabra)
    return aux


def dejar_uno_solo(ayuda):
    piti = []
    for hola in ayuda:
        for i in hola:
            piti.append(i)
    return piti


# Cargamos las coversaciones del archivo de texto
ruta = "archivo_chico.txt"
corpus = cargar_datos(ruta)

# Creamos un diccionario con la lista de conversaciones
conversaciones = { i : corpus[i] for i in range(0, len(corpus) ) }

# Separamos las preguntas de las respuestas
preguntas , respuestas = separar(conversaciones)

# Limpieza de datos
# Normalizamos las preguntas y las respuestas
preguntas_normalizadas = [limpiar_datos(pregunta) for pregunta in preguntas]
respuestas_normalizadas = [limpiar_datos(respuesta) for respuesta in respuestas]

# Tokenizamos
tokenized_preguntas = tokeniza(preguntas_normalizadas)
tokenized_respuestas = tokeniza(respuestas_normalizadas)

# Removemos las stopwords
without_stopwords_preguntas = remover_stopWords(tokenized_preguntas)
without_stopwords_respuestas = remover_stopWords(tokenized_respuestas)







