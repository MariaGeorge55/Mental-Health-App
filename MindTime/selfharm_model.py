import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import re, string, random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
"""
from keras.layers import Input, Conv2D, Activation, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import AveragePooling1D

scaler = StandardScaler()

from keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.lite as lite
"""
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


def categorizable(data):
       from sklearn.preprocessing import  LabelEncoder
       return LabelEncoder().fit_transform(data)


def categorizable2(data):
    from sklearn.preprocessing import MultiLabelBinarizer
    return MultiLabelBinarizer().fit_transform(data)

def load_data():
    data = pd.read_csv("selfharm.csv")
    data = data[["label", "content"]]
    predict = "label"
    x = np.array(data.drop([predict], 1))  ##x holds content 1d array
    y = np.array(data[predict])  ##y holds label class 1d array
    return x,y

def tokenizedata(x):
    stop_words = stopwords.words('english')
    cleaned_tokens_list = []
    tweet_tokens = []
    for sentence in x:
        for word in sentence:
            tweet_tokens.append(word_tokenize(word))

    for tokens in tweet_tokens:
        cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    return cleaned_tokens_list

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def NaiveBayes_accuracy(cleaned_tokens_list):
    selfharm_tweets = cleaned_tokens_list[:404]
    neutral_tweets = cleaned_tokens_list[404:]
    selfharm_tweets_model = get_tweets_for_model(selfharm_tweets)
    neutral_model = get_tweets_for_model(neutral_tweets)
    selfharm_tweets_dataset = [(tweet_dict, "Selfharm")
                               for tweet_dict in selfharm_tweets_model]
    neutral_dataset = [(tweet_dict, "Neutral")
                       for tweet_dict in neutral_model]
    dataset = selfharm_tweets_dataset + neutral_dataset
    random.shuffle(dataset)
    train_data = dataset[:500]
    test_data = dataset[500:]
    classifier = NaiveBayesClassifier.train(train_data)
    print("Accuracy of NaiveBayesClassifier is:", classify.accuracy(classifier, test_data))

def split(cleaned_tokens_list,y):
    x = categorizable2(cleaned_tokens_list)
    y = categorizable(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=109)
    return X_train, X_test, y_train, y_test


def SVM_linear(X_train, y_train,X_test, y_test):
    res = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    linear_pred = res.predict(X_test)
    accuracy_lin = res.score(X_test, y_test)
    print("Accuracy Linear Kernel:", accuracy_lin)

    svclassifier = svm.SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    accuracy_rbf = svclassifier.score(X_test, y_test)
    print("Accuracy rbf Kernel:", accuracy_rbf)

    svclassifier = svm.SVC(kernel='sigmoid')
    svclassifier.fit(X_train, y_train)
    accuracy_sig = svclassifier.score(X_test, y_test)
    print("Accuracy sigmoid Kernel:", accuracy_sig)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    accuracy_knn = knn.score(X_test, y_test)
    print("Accuracy KNN :", accuracy_knn)


def getClassesNum(y):
    classes=set(y)
    print(classes.__len__())
    return classes.__len__()


def construct_cnn(x,y):
    model = Sequential()
    """
    model.add(Conv2D(12, (2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(getClassesNum(y), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
     metrics=['accuracy'], use_landmarks=True,
                  use_hog_and_landmarks=True,
                  use_hog_sliding_window_and_landmarks=True,
                  use_batchnorm_after_conv_layers=True,
                  use_batchnorm_after_fully_connected_layers=False)

    input_layer = Dense(32, input_shape=(8,))
    model.add(input_layer)
    hidden_layer = Dense(64, activation='relu');
    model.add(hidden_layer)
    output_layer = Dense(8)
    model.add(output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    model.save('E:/CNNHarm/model')
"""
    model = Sequential()
   # x = np.array(x)
    #x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    scaler.fit(x)
    x = scaler.transform(x)

    model.add(LSTM(22, input_shape=(x[1],1), return_sequences=True, implementation=2))
    model.add(TimeDistributed(Dense(1)))
    #model.add(AveragePooling1D())

    model.add(Flatten())

    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    model.save('E:/CNNHarm/model')

    return model


def init_callbacks():
    from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    base_path = 'E:/CNNHarm/model_weights/'
    trained_models_path = base_path + 'model_weights'
    model_names = trained_models_path + '.{epoch:04d}--{val_loss:.4f}--{val_accuracy:.4f}.h5'
    model_checkpoint = ModelCheckpoint(model_names, monitor = 'val_accuracy', verbose=1,save_best_only=True)
    early_stopping= EarlyStopping(patience=20)
    callbacks = [model_checkpoint,early_stopping]
    return callbacks


def tarin_model_svmandNB():
    x, y = load_data()
    cleaned_tokens_list = tokenizedata(x)
    NaiveBayes_accuracy(cleaned_tokens_list)
    X_train, X_test, y_train, y_test=split(cleaned_tokens_list,y)
    SVM_linear(X_train, y_train, X_test, y_test)

batch_size = 32
epochs=100
def train_model():
    x, y = load_data()
    cleaned_tokens_list = tokenizedata(x)
    x_train, x_test, y_train, y_test  = split(cleaned_tokens_list,y)
    y = categorizable(y)
    x = categorizable2(x)

    model = construct_cnn(x,y)
    model.fit(x_train, y_train, epochs=100, batch_size=6000, verbose=1, validation_data=(x_test, y_test))

    #model.fit( x_train, y_train, batch_size = batch_size, epochs = epochs, verbose=1, callbacks= init_callbacks(), validation_data=(x_test,y_test))

#train_model()
tarin_model_svmandNB()