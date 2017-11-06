import gensim.models.word2vec as word2vec
import re

filename = '../Total_Data/03Data/HarryPotter.txt'
f=open(filename,'r')
file_read=f.read()
words_=re.sub("[^a-zA-Z]+", " ",file_read).lower() #正则匹配,只留下单词，且大写改小写

model = word2vec.Word2Vec(sentences=words_,size=100, window=5, min_count=5, workers=4)

print(model.wv)
