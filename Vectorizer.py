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
cv = CountVectorizer(ngram_range=(1,1), stop_words = stopwords, max_features = 50)
sparsearray = cv.fit_transform(tweettextonly)
vocab = cv.vocabulary_
count_values = sparsearray.toarray().sum(axis=0)

unigrams = []
for ug_count, ug_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True):
    unigrams.append(ug_text)
print(unigrams)

#bigrams
cv = CountVectorizer(ngram_range=(2,2), stop_words = stopwords, max_features = 50)
sparsearray = cv.fit_transform(tweettextonly)
vocab = cv.vocabulary_
count_values = sparsearray.toarray().sum(axis=0)

bigrams = []
for bg_count, bg_text in sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True):
    bigrams.append(bg_text)
print(bigrams)

#trigrams
cv = CountVectorizer(ngram_range=(3,3), stop_words = stopwords, max_features = 50)
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
    
    
    tokens = tweet.split(" ")
    
    # calculate the frequency of uni-gram
    for ug in unigrams:
        count = 0
        if ug in tweet:
            for token in tokens :
                if (ug == token):
                    count = count + 1
        uni.append(count)
    vec.append(uni)
    
    # calculate the frequency of bi-gram
    for bg in bigrams:
        count = 0
        if bg in tweet:
            for i in range(0, len(tokens)-1):
                bigram = tokens[i] + " " + tokens[i+1]
                if (bg == bigram):
                    count = count + 1
        bi.append(count)
    vec.append(bi)
    
    # calculate the frequency of tri-gram
    for tg in trigrams:
        count = 0
        if tg in tweet:
            for i in range(0, len(tokens)-2):
                trigram = tokens[i] + " " + tokens[i+1] + " " + tokens[i+2]
                if (tg == trigram):
                    count = count + 1
        tri.append(count)
    vec.append(tri)
    
    result.append(vec)
    
print(result)
