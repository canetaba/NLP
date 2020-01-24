from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

# Prediction de sentimientos
# Preparar los datos
# Se restringirán las revisiones de peliculas a las 10000 palabras mas comunes
# Se cortaran las revisiones despues de solo 20 palabras
# La red aprendera embeddings de dimension 8 por cada 10000 palabras
# transformara las secuencias enteras 2D en sequencias embebidas (3D float tensor)
# aplanara el tensor a 2D y entrenara una red simple de 1 capa para clasificacion

max_features = 10000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)



model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())

model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)
