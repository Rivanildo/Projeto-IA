import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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
x_test = vectorizer.transform(test.Comment)

# classificador do pacote naive bayes
classificador = MultinomialNB()


# Validacao cruzada
scores = cross_val_score(classificador, x_train, train.Insult, cv = 10, scoring="roc_auc")

# Treinar o naive bayes
model = classificador.fit(x_train, list(train.Insult))

#Classifica os tests
predictions = model.predict_proba(x_test)[:,1]


# gravando os resultados no arquivo
submission = pd.DataFrame({'id': test.id, 'insult': predictions})
submission.to_csv('resultsMultinomialCrossValidation.csv', index=False)