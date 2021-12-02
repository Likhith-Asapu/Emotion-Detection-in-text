import nltk
from sklearn.feature_extraction.text import CountVectorizer

tweettextonly = [
        ' I am happy loooool', 
        ' I LOVE YOU', 
        ' I am happy our mai khush hu', 
        ' I am very happy', 
        ' I am happy', 
        ' I am happy', 
        ' I am happy', 
        ' I am happy', 
        ' I am happy', 
        ' I am Sad', 
        ' I am Sad', 
        ' I am Sad', 
        ' I am Sad', 
        ' I am angry', 
        ' I AM ANGRY', 
        ' I AM ANGRY', 
        ' I AM ANGRY'
    ]

stopwords = nltk.corpus.stopwords.words('english')

# unigrams
cv = CountVectorizer(ngram_range=(1,1), stop_words = stopwords, max_features = 10)
sparsearray = cv.fit_transform(tweettextonly)
vocab = cv.vocabulary_
count_values = sparsearray.toarray().sum(axis=0)

unigrams = []
for ug_count, ug_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True):
    unigrams.append(ug_text)
print(unigrams)

#bigrams
cv = CountVectorizer(ngram_range=(2,2), stop_words = stopwords, max_features = 10)
sparsearray = cv.fit_transform(tweettextonly)
vocab = cv.vocabulary_
count_values = sparsearray.toarray().sum(axis=0)

bigrams = []
for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True):
    bigrams.append(bg_text)
print(bigrams)

#trigrams
cv = CountVectorizer(ngram_range=(3,3), stop_words = stopwords, max_features = 10)
sparsearray = cv.fit_transform(tweettextonly)
vocab = cv.vocabulary_
count_values = sparsearray.toarray().sum(axis=0)

trigrams = []
for tg_count, tg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True):
    trigrams.append(tg_text)
print(trigrams)

print()

result = []
for tweet in tweettextonly:
    vec = []
    
    uni = []
    bi = []
    tri = []
    
    for ug in unigrams:
        if ug in tweet:
            uni.append(1)
        else :
            uni.append(0)
    vec.append(uni)
    
    for bg in bigrams:
        if bg in tweet:
            bi.append(1)
        else :
            bi.append(0)
    vec.append(bi)
    
    for tg in trigrams:
        if tg in tweet:
            tri.append(1)
        else :
            tri.append(0)
    vec.append(tri)
    
    result.append(vec)
    
print(result)
