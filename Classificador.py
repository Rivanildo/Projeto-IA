import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score


# Leitura dos arquivos csv
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Convert a collection of text documents to a matrix of token counts
# Conta a frequencia de palavras em um texto
vectorizer = CountVectorizer()

# Learn the vocabulary dictionary and return term-document matrix.
# treinando o algoritmo com a funcao fit_transform
x_train = vectorizer.fit_transform(train.Comment)
x_test = vectorizer.fit_transform(test.Comment)





