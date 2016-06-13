import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Leitura dos arquivos csv
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Convert a collection of text documents to a matrix of token counts
# Conta a frequencia de palavras em um texto
vectorizer = CountVectorizer()

# Learn the vocabulary dictionary and return term-document matrix.
# treinando o algoritmo com a funcao fit_transform
x_train = vectorizer.fit_transform(train.Comment)
x_test = vectorizer.transform(test.Comment)

# classificador do pacote naive bayes
classificador = MultinomialNB()

# Treinar o naive bayes
# treina o classificador passando a matriz de treino e os rotulos (insultos)
classificador.fit(x_train,train.Insult)

# classifica as frases dos arquivos de test
predictions = classificador.predict(x_test)


# gravando os resultados no arquivo submission.csv
submission = pd.DataFrame({'id': test.id, 'insult': predictions})
submission.to_csv('submission.csv', index=False)