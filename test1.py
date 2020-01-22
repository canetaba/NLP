import pandas as pd
import regex as re
import nltk
import numpy as np
import logging # verbosidad (útil para saber qué está ocurriendo al ejecutar código)
from nltk.chat.util import Chat, reflections



nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('spanish')

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


def limpiar_datos(text):
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
    # Remueve las stopwords
    aux = []
    for palabra in arreglo:
        aux_stopwords_palabra = [word for word in palabra if word not in stopwords]
        aux.append(aux_stopwords_palabra)
    return aux

ruta = "archivo_chico.txt"
sentences = cargar_datos(ruta)
print(sentences)

# Creamos un diccionario con la lista de conversaciones
conversaciones = { i : sentences[i] for i in range(0, len(sentences) ) }


preguntas = []
respuestas = []
for i in range(0, len(conversaciones)):
    m = i
    if m%2:
        respuestas.append(conversaciones.get(i))
    else:
        preguntas.append(conversaciones.get(i))

print ("Preguntas: ", preguntas)
print("Respuestas ", respuestas)



# Limpiamos los datos
# Normalizamos las preguntas y las respuestas
preguntas_normalizadas = [limpiar_datos(pregunta) for pregunta in preguntas]
#print(preguntas_normalizadas)

respuestas_normalizadas = [limpiar_datos(respuesta) for respuesta in respuestas]
#print(respuestas_normalizadas)

# Tokenizamos
tokenized_preguntas = tokeniza(preguntas_normalizadas)
tokenized_respuestas = tokeniza(respuestas_normalizadas)
# print(tokenized_preguntas)
# print(tokenized_respuestas)

# Removemos las stopwords
without_stopwords_preguntas = []
without_stopwords_respuestas = []

without_stopwords_preguntas = remover_stopWords(tokenized_preguntas)
without_stopwords_respuestas = remover_stopWords(tokenized_respuestas)


print (without_stopwords_preguntas)
print (without_stopwords_respuestas)
