from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# Crea un tokenizador, configurado para tomar en cuenta las 1000
# palabras mas comunes
tokenizer = Tokenizer(num_words=1000)

#Contruye el indice de palabras
tokenizer.fit_on_texts(samples)

# Transforma strings en una lista de indices enteros
sequences = tokenizer.texts_to_sequences(samples)

# Tokeniza
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index
print('Found %s unique tokens. ' % len(word_index))