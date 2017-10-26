from nltk import ngrams
sentence = 'Maybe i should get some coffee before starting'
n = 2
sixgrams = ngrams(sentence.split(), n)
for grams in sixgrams:
  print(grams)