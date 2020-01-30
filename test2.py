import regex as re
import nltk
import pandas as pd
import tensorflow as tf
import numpy as np


def cargar_datos(archivo):
    # Carga los archivos en un arreglo
    oraciones = []
    with open(ruta, encoding='utf-8', errors='ignore') as file:  # Usamos utf-8 para preservar los caracteres especiales
        for line in file:
            oraciones.append((line.rstrip()))  # Usamos la función strip para remover el caracter \n (nueva línea)
    return oraciones


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


def tokeniza(sentencias):
    # Tokenizamos las palabras
    arreglo = []
    arreglo = [nltk.word_tokenize(sentencia) for sentencia in sentencias]
    return arreglo


def remove_stop_words(corpus):
    stop_words = ['de', 'para', 'el', 'ella','la',
                 'los', 'las', 'es', 'lo','o','a',
                 'y','ya','le','que', 'una', 'un',
                 'con','se','por','mi','del','te','me','al','tu',
                 'en','su','pero', 'entre', 'sin', 'embargo',
                 'ni','les','nos','mas','ey','aj','oh','es']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))

    return results


ruta = "archivo_chico.txt"
corpus = cargar_datos(ruta)

corpus_limpio = [limpiar_datos(cuerpo) for cuerpo in corpus]
print(corpus_limpio)

corpus_limpio = remove_stop_words(corpus_limpio)
print(corpus)

words = []
for text in corpus_limpio:
    words += text.split(' ')
words = set(words)

print(words)

word2int = {}

for i,word in enumerate(words):
    word2int[word] = i

sentences = []
for sentence in corpus_limpio:
    sentences.append(sentence.split())

print(sentences)

WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        # Acá se toma una ventana de -WINDOWS_SIZE, WINDOWS_SIZE para generar el skip gram. Dado que las
        # frases son cortas, se utiliza min y max para tener cuidado con los límites de la frase.
        # Además, el +1 en el límite superior es para considerar el índice de la propia palabra en cuestión
        # (probar qué ocurre cuando se elimina dicho +1)
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])

print(data)

df = pd.DataFrame(data, columns = ['input', 'label'])
print(df.head(10))
print(df.shape)
print(word2int)


# Creacion de un grafo de Tensorflow
# En esta etapa queremos convertir el conjunto de palabras en un
# grafo tensor, con el objetivo siguiente de poder entrenar una red y predecir
# una respuesta

ONE_HOT_DIM = len(words)

# Función para transformar números enteros a one hot encodings.
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = [] # Palabra de entrada
Y = [] # Palabra objetivo

for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))

# Convertir a array de NumPy
X_train = np.asarray(X)
Y_train = np.asarray(Y)

print(X_train[:10], Y_train[:10])

# Definición de placeholders para X_train e Y_train
x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

# Encaje (embedding) será de dimensión 2, para así visualizar fácilmente los resultados
EMBEDDING_DIM = 2

# Capa oculta: eventualmente, representará el vector de palabras
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM])) # pesos
b1 = tf.Variable(tf.random_normal([1])) # sesgos
hidden_layer = tf.add(tf.matmul(x,W1), b1) # x*W1 + b1

# output layer
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2), b2)) # softmax(hidden*W2 + b2)

# loss function: cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# training operation
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 20000
for i in range(iteration):
    # Entrada es X_train, correspondiente a una palabra en formato one hot.
    # Etiqueta es Y_train, correspondiente a una palabra vecina también en formato one hot.
    # Entrenamiento NO ES SIMULTÁNEO con varias palabras a la vez
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 3000 == 0:
        print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))


# Vectores de palabras
# Ahora la capa oculta (W1 + b1) se convierte en la tabla de búsqueda de palabras
vectors = sess.run(W1 + b1) # Técnicamente, no es necesario añadir b1, ya que es una constante para todos los pesos
print(vectors)


