import sklearn
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import numpy as np
import re, string, random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.neighbors import KNeighborsClassifier
def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def categorizable(data):
       from sklearn.preprocessing import  LabelEncoder
       return LabelEncoder().fit_transform(data)


def categorizable2(data):
    from sklearn.preprocessing import MultiLabelBinarizer
    return MultiLabelBinarizer().fit_transform(data)

data = pd.read_csv("empty.csv")
data = data[["label", "content"]]
predict = "label"
x = np.array(data.drop([predict],1))  ##x holds content 1d array
y = np.array(data[predict])    ##y holds label class 1d array

#neutral_tweets = data["content"][405:]
stop_words = stopwords.words('english')

cleaned_tokens_list=[]
tweet_tokens = []
for sentence in x:
       for word in sentence:
        tweet_tokens.append(word_tokenize(word))

for tokens in tweet_tokens:
       cleaned_tokens_list.append(remove_noise(tokens, stop_words))

empty_tweets = cleaned_tokens_list[:826]
neutral_tweets = cleaned_tokens_list[826:1500]
empty_tweets_model = get_tweets_for_model(empty_tweets)
neutral_model = get_tweets_for_model(neutral_tweets)

empty_tweets_dataset = [(tweet_dict, "Empty")
                 for tweet_dict in empty_tweets_model]

neutral_dataset = [(tweet_dict, "Neutral")
                 for tweet_dict in neutral_model]

dataset = empty_tweets_dataset + neutral_dataset
random.shuffle(dataset)
train_data = dataset[:1200]
test_data = dataset[1200:]
classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy of NaiveBayesClassifier is:", classify.accuracy(classifier, test_data))

#custom_tweet = " My life is meaningless"  # self-harm
custom_tweet = " I went to school yesterday , I hate my friends. "                     #Anger

custom_tokens = remove_noise(word_tokenize(custom_tweet))
print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))

x = categorizable2(cleaned_tokens_list)
y = categorizable(y)

#print(y)
#plt.plot(x,y)
#plt.show()
X_train, X_test, y_train, y_test = train_test_split(x,y , train_size=0.7, random_state=109)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
accuracy_knn = knn.score(X_test, y_test)
print("Accuracy KNN :", accuracy_knn)

res = svm.SVC(kernel='linear').fit(X_train,y_train )
linear_pred = res.predict(X_test)
accuracy_lin = res.score(X_test, y_test)
print("Accuracy Linear Kernel:", accuracy_lin)

svclassifier = svm.SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
accuracy_rbf = svclassifier.score(X_test, y_test)
print("Accuracy rbf Kernel:", accuracy_rbf)

svclassifier = svm.SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)
accuracy_sig = svclassifier.score(X_test, y_test)
print("Accuracy sigmoid Kernel:", accuracy_sig)

"""
svclassifier = svm.SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)
accuracy_poly = svclassifier.score(X_test, y_test)
print("Accuracy poly Kernel:", accuracy_poly)
"""
