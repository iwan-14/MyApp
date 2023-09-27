from flask import Flask, render_template, redirect, request
from scraper import Scraper
import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import string
import numpy as np
import os
import json
from nltk import word_tokenize
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import svm
import csv
import time
import swifter
import pickle
import json
import googletrans
from googletrans import Translator
from googletrans import LANGUAGES
import httpx
from textblob import TextBlob


app = Flask(__name__)
app.secret_key = 'hellow'
#Upload CSV
ALLOWED_EXTENSION = set(['csv'])
app.config['UPLOAD_FOLDER']='uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.route('/', methods=['POST', 'GET'])
def main():
    return redirect('index')

@app.route('/index', methods=['POST', 'GET'])
def dashboard():
    text = pd.read_csv('uploads/Dataclean.csv', sep=',', encoding='latin1')
    positif, negatif, netral= text['Sentiment'].value_counts()
    total = positif + negatif + netral
    neutral  = (sum(text['Sentiment']=='neutral'))
    positive = (sum(text['Sentiment']=='positive'))
    negative = (sum(text['Sentiment']=='negative'))

    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    # Open the CSV file
    with open('uploads/Dataclean.csv', 'r', encoding='latin1') as file:
    # Create a CSV reader object
        reader = csv.reader(file)
    
        # Iterate over the rows of the file
        for row in reader:
            # row is a list of values in the row
            Sentiment = row[1]  # Assume the first column is the label
            # Count the number of positive, negative, and neutral labels
            if Sentiment == 'positive':
                positive_count += 1
            elif Sentiment == 'negative':
                negative_count += 1
            elif Sentiment == 'neutral':
                neutral_count += 1

    # Dump the counts to a JSON object
    counts = {
    'positive': positive_count,
    'negative': negative_count,
    'neutral': neutral_count
    }
    with open('counts.json', 'w') as file:
        json.dump(counts, file)

    
    with open("counts.json") as f:
        data = json.load(f)

    pos = data["positive"]
    neg = data["negative"]
    neu = data["neutral"]


    return render_template('index.html', total=total, positif=positive, negatif=negative, netral=neutral, pos1=pos,
                           neg1=neg, neu1=neu)

@app.route('/help', methods=['POST', 'GET'])
def help():
    return render_template('help.html')

@app.route('/about', methods=['POST', 'GET'])
def about():
    return render_template('about.html')

@app.route('/uploaddata', methods=['GET', 'POST'])
def uploaddata():
    if request.method == 'GET':
        return render_template('uploaddata.html')
    
    elif request.method == 'POST':
        file = request.files['file']
        
        if 'file' not in request.files:
            return redirect(request.url)

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file.filename = "dataset.csv"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

            data3 = []
            
            with open ('uploads/dataset.csv', encoding='latin1') as f:
                reader = csv.DictReader(f)
                
                [data3.append(dict(row)) for row in reader]
            
            return render_template('uploaddata.html', data=data3, list=list, len=len)
        
@app.route('/scrape', methods=['POST', 'GET'])
def search():
    return render_template('scrape.html')

@app.route('/scraper', methods=['POST', 'GET'])
def scraper():
    start_time = time.time()
    keyword = request.args.get('keyword')
    tweets = request.args.get('tweets')
    sa = Scraper()
    status = "Scraping Completed"
    
    tweet_list = sa.scraping(keyword, tweets)
    
    fa = 'Data has been saved'
    
    df = pd.DataFrame(tweet_list, columns=['Datetime', 'Tweet Id', 'Tweet', 'Lang', 'Username'])
    print(df.head())
    
    df.to_csv('uploads/datatest.csv', index=False)
    
    elapsed_time = time.time() - start_time
    elapsed_time_minutes = elapsed_time / 60
    print(f'Elapsed Time: {elapsed_time_minutes:.2f} minute')
    
    data2 = []
    
    with open ('uploads/datatest.csv', encoding='latin1') as f:
        reader = csv.DictReader(f)
        
        [data2.append(dict(row)) for row in reader]
    
    return render_template('scrape.html', data=data2, list=list, len=len, fa=fa, df1=f, status=status)

@app.route('/scraperlabel', methods=['POST', 'GET'])
def scraperlabel():
    start_time = time.time()
    keyword = request.args.get('keyword')
    tweets = request.args.get('tweets')
    sa = Scraper()
    status = "Scraping Completed"
    
    tweet_list = sa.scraping(keyword, tweets)
    
    fa = 'Data has been saved'
    
    df = pd.DataFrame(tweet_list, columns=['Datetime', 'Tweet Id', 'Tweet', 'Lang', 'Username'])
    print(df.head())
    
    timeout = httpx.Timeout(5) # 5 seconds timeout
    translator = Translator(timeout=timeout)
    
    df['Translated'] = df['Tweet'].apply(lambda x: translator.translate(x, dest='en').text)
    
    polarity = lambda x: TextBlob(x).sentiment.polarity

    df['Sentiment'] = df['Translated'].apply(polarity)
    df
    
    def analysis(score):
        if score > 0:
            return ('positif')
        elif score ==0:
            return ('netral')
        else:
            return ('negatif')
        
    df['Sentiment'] = df['Sentiment'].apply(analysis)
    
    df.to_csv('uploads/datatest.csv', index=False)
    
    elapsed_time = time.time() - start_time
    elapsed_time_minutes = elapsed_time / 60
    print(f'Elapsed Time: {elapsed_time_minutes:.2f} minute')
    
    data2 = []
    
    with open ('uploads/datatest.csv', encoding='latin1') as f:
        reader = csv.DictReader(f)
        
        [data2.append(dict(row)) for row in reader]
    
    return render_template('scrape.html', data=data2, list=list, len=len, fa=fa, df1=f, status=status)

@app.route('/preprocessing', methods=['POST', 'GET'])
def preprocessing():
    return render_template('preprocessing.html')

@app.route('/preprocessingdata', methods=['POST', 'GET'])
def normal():
    start_time = time.time()
    df = pd.read_csv('uploads/dataset.csv', sep=',', encoding='latin1')
    df.drop_duplicates(subset='Tweet', keep=False, inplace=True)

    # ------ Proses Case Folding --------
    # gunakan fungsi Series.str.lower() pada Pandas
    df['Casefolding'] = df['Tweet'].str.lower()

    def remove_tweet_special(text):
        # Remove Emoticon
        text = text.encode('ascii', 'replace').decode('ascii')
        # Remove additional code
        text = text.replace("\\xe2\\x80\\xa6", "")
        # Convert www.* or https?://* to URL
        text = re.sub(r"http\S+", "", text)
        # Convert @username to AT_USER
        text = re.sub("@[^\s]+", "", text)
        # Remove additional white spaces
        text = re.sub("[\s]+", " ", text)
        # Replace #word with word
        text = re.sub(r"#([^\s]+)", r"\1", text)
        # Menghapus angka dari teks
        text = re.sub(r"\d+", "", text)
        # Menganti tanda baca dengan spasi
        text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        return text
                    
    df['Cleansing'] = df['Casefolding'].apply(remove_tweet_special)

    # NLTK word rokenize 
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    df['Tokenizing'] = df['Cleansing'].apply(word_tokenize_wrapper)

    list_stopwords = stopwords.words('indonesian', 'english')


    # ---------------------------- manualy add stopword  ------------------------------------
    # append additional stopword
    list_stopwords.extend(['jg', 'josss', "yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                        'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'wkwk', 'wkwkwk', 'mjb', 'tbh',
                        'knl', 'teruz', 'zeh', 'tra', 'kks', 'hhahahahahahahahahahahahahahahahahahahahaha',
                        'buahahahahahahahahahahahahahahahahahahahahahahahah', 'mlh', 'pye', 'tauu', 'bjo',
                        'wkwksjsjsksksjjss', 'jis', 'ckckckckckk', 'hrus', 'sgera', 'adl', 'zzz', 'yagaksi',
                        'lg', 'udh', 'dr', 'duarr', 'gw', 's', 'cm'
                        ])

    list_stopwords.extend(["stopwords"][0].split(' '))

    list_stopwords = set(list_stopwords)

    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    df['Stopword'] = df['Tokenizing'].apply(stopwords_removal) 

    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in df['Stopword']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '
                
    print(len(term_dict))
    print("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])
        
    print(term_dict)
    print("------------------------")


    # apply stemmed term to dataframe
    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    df['Stemming'] = df['Stopword'].swifter.apply(get_stemmed_term)

    df['Dataclean'] = df['Stemming'].apply(lambda x: ' '.join(x))
    
    df.drop_duplicates(subset='Tweet', keep=False, inplace=True)

    df.to_csv('uploads/Dataclean.csv', sep=',', encoding='latin1', index=False)

    print(df)
    
    elapsed_time = time.time() - start_time
    elapsed_time_minutes = elapsed_time / 60
    print(f'Elapsed Time: {elapsed_time_minutes:.2f} minute')
    
    data = []
    
    with open ('uploads/Dataclean.csv', encoding='latin1') as f:
        reader = csv.DictReader(f)
        
        [data.append(dict(row)) for row in reader]
    
    return render_template('preprocessing.html', data=data, list=list, len=len)

@app.route('/classification', methods=['POST', 'GET'])
def classification():
    return render_template('klasifikasisvm.html')

@app.route('/svm', methods=['POST', 'GET'])
def svm():
    start_time = time.time()
    # load the data into a DataFrame
    df = pd.read_csv("uploads/Dataclean.csv", sep=',', encoding='latin1')

    # extract the tweet text and the labels
    text = df["Stemming"].tolist()
    labels = df["Sentiment"].tolist()
    
    # create the transform
    vectorizer = TfidfVectorizer()

    # tokenize and build vocab
    vectorizer.fit(text)

    # encode document
    X = vectorizer.transform(text)
    
    # scores = X.toarray()
    # print(scores)
    
    # feature_names = vectorizer.get_feature_names()

    # create dataframe with Tfidf scores
    # df = pd.DataFrame(scores, columns=feature_names)

    # print dataframe
    # print(df)

        
    # split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=0)

    # train an SVM model on the training data
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    modelrbf = SVC(kernel="rbf")
    modelrbf.fit(X_train, y_train)
    y_predrbf = modelrbf.predict(X_test)

    from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, classification_report
        
    print (classification_report(y_test, y_pred))

    print (classification_report(y_test, y_predrbf))

    f1_score1 = f1_score(y_test, y_pred, average='macro')
    accuracy_score1 = accuracy_score(y_test, y_pred)
    precision_score1 = precision_score(y_test, y_pred, average='macro')
    recall_score1 = recall_score(y_test, y_pred, average='macro')

    print(f1_score1)
    print(accuracy_score1)
    print(precision_score1)
    print(recall_score1)

    f1_score2 = f1_score(y_test, y_predrbf, average='macro')
    accuracy_score2 = accuracy_score(y_test, y_predrbf)
    precision_score2 = precision_score(y_test, y_predrbf, average='macro')
    recall_score2 = recall_score(y_test, y_predrbf, average='macro')

    print(f1_score2)
    print(accuracy_score2)
    print(precision_score2)
    print(recall_score2)
    
    predictions = np.array(y_test)
    ground_truth = np.array(y_pred)
    
    predictions1 = np.array(y_test)
    ground_truth1 = np.array(y_predrbf)

    # Calculate the confusion matrix
    confusion_mat_linear = confusion_matrix(ground_truth, predictions)
    confusion_mat_rbf = confusion_matrix(ground_truth1, predictions1)
    
    print(confusion_mat_linear)
    print(confusion_mat_rbf)


    
    elapsed_time = time.time() - start_time
    elapsed_time_minutes = elapsed_time / 60
    print(f'Elapsed Time: {elapsed_time_minutes:.2f} minute')
        
    return render_template ('klasifikasisvm.html', f1_score1=f1_score1, accuracy_score1=accuracy_score1, precision_score1=precision_score1,
                            recall_score1=recall_score1, f1_score2=f1_score2, accuracy_score2=accuracy_score2, precision_score2=precision_score2,
                            recall_score2=recall_score2, confusion_mat=confusion_mat_linear, confusion_mat1=confusion_mat_rbf)

@app.route('/prediksi', methods=['POST', 'GET'])
def tesmodel():
    return render_template('prediksisvm.html')

@app.route('/prediksisvm', methods=['POST', 'GET'])
def modeltes():
    keyword = request.args.get('keyword')
    tweets = request.args.get('tweets')
    sa = Scraper()
    
    tweet_list = sa.scraping(keyword, tweets)
    
    df = pd.DataFrame(tweet_list, columns=['Datetime', 'Tweet Id', 'Tweet', 'Lang', 'Username'])
    df = pd.DataFrame(df[['Datetime', 'Tweet']])
    
    df['Casefolding'] = df['Tweet'].str.lower()

    
    def remove_tweet_special(text):
        # Remove Emoticon
        text = text.encode('ascii', 'replace').decode('ascii')
        # Remove additional code
        text = text.replace("\\xe2\\x80\\xa6", "")
        # Convert www.* or https?://* to URL
        text = re.sub(r"http\S+", "", text)
        # Convert @username to AT_USER
        text = re.sub("@[^\s]+", "", text)
        # Remove additional white spaces
        text = re.sub("[\s]+", " ", text)
        # Replace #word with word
        text = re.sub(r"#([^\s]+)", r"\1", text)
        # Menghapus angka dari teks
        text = re.sub(r"\d+", "", text)
        # Menganti tanda baca dengan spasi
        text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        return text
                    
    df['Remove_Special_Text'] = df['Casefolding'].apply(remove_tweet_special)

    # NLTK word rokenize 
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    df['Tokenizing'] = df['Remove_Special_Text'].apply(word_tokenize_wrapper)

    list_stopwords = stopwords.words('indonesian', 'english')


    # ---------------------------- manualy add stopword  ------------------------------------
    # append additional stopword
    list_stopwords.extend(['jg', 'josss', "yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                        'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'wkwk', 'wkwkwk', 'mjb', 'tbh',
                        'knl', 'teruz', 'zeh', 'tra', 'kks', 'hhahahahahahahahahahahahahahahahahahahahaha',
                        'buahahahahahahahahahahahahahahahahahahahahahahahah', 'mlh', 'pye', 'tauu', 'bjo',
                        'wkwksjsjsksksjjss', 'jis', 'ckckckckckk', 'hrus', 'sgera', 'adl', 'zzz', 'yagaksi',
                        'lg', 'udh', 'dr', 'duarr', 'gw', 's', 'cm'
                        ])

    list_stopwords.extend(["stopwords"][0].split(' '))

    list_stopwords = set(list_stopwords)

    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    df['Stopword'] = df['Tokenizing'].apply(stopwords_removal) 

    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in df['Stopword']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '
                
    print(len(term_dict))
    print("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])
        
    print(term_dict)
    print("------------------------")

    # apply stemmed term to dataframe
    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    df['Stemming'] = df['Stopword'].swifter.apply(get_stemmed_term)
    
    df['Dataclean'] = df['Stemming'].apply(lambda x: ' '.join(x))
    
    df.to_csv('uploads/modeltest.csv', index=False)
    
    df = pd.read_csv("uploads/Dataclean.csv", sep=',', encoding='latin1')

    # extract the tweet text and the labels
    text = df["Tweet"].tolist()
    labels = df["Sentiment"].tolist()
    
    # create the transform
    vectorizer = TfidfVectorizer()

    # tokenize and build vocab
    vectorizer.fit(text)

    # encode document
    X = vectorizer.transform(text)
    
    # split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=0)

    # train an SVM model on the training data
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # load the new data into a DataFrame
    df_new = pd.read_csv("uploads/modeltest.csv", sep=',', encoding='latin1')

    # extract the tweet text
    text_new = df_new["Tweet"].tolist()

    # encode the new data
    X_new = vectorizer.transform(text_new)

    # make predictions using the trained SVM
    predictions = model.predict(X_new)

    print(predictions)

    df_new["Prediction"] = predictions
    
    df_new = pd.DataFrame(df_new[['Datetime', 'Tweet', 'Prediction']])
    
    df_new.to_csv('uploads/predicted.csv', index=False, encoding='latin1')
    
    data5 = []
    
    with open ('uploads/predicted.csv', encoding='latin1') as f:
        reader = csv.DictReader(f)
        
        [data5.append(dict(row)) for row in reader]
    
    return render_template('prediksisvm.html', data=data5, list=list, len=len)

@app.route('/prediksisvmrbf', methods=['POST', 'GET'])
def modeltesrbf():
    keyword = request.args.get('keyword')
    tweets = request.args.get('tweets')
    sa = Scraper()
    
    tweet_list = sa.scraping(keyword, tweets)
    
    df = pd.DataFrame(tweet_list, columns=['Datetime', 'Tweet Id', 'Tweet', 'Lang', 'Username'])
    df = pd.DataFrame(df[['Datetime', 'Tweet']])
    
    df['Casefolding'] = df['Tweet'].str.lower()

    
    def remove_tweet_special(text):
        # Remove Emoticon
        text = text.encode('ascii', 'replace').decode('ascii')
        # Remove additional code
        text = text.replace("\\xe2\\x80\\xa6", "")
        # Convert www.* or https?://* to URL
        text = re.sub(r"http\S+", "", text)
        # Convert @username to AT_USER
        text = re.sub("@[^\s]+", "", text)
        # Remove additional white spaces
        text = re.sub("[\s]+", " ", text)
        # Replace #word with word
        text = re.sub(r"#([^\s]+)", r"\1", text)
        # Menghapus angka dari teks
        text = re.sub(r"\d+", "", text)
        # Menganti tanda baca dengan spasi
        text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        return text
                    
    df['Remove_Special_Text'] = df['Casefolding'].apply(remove_tweet_special)

    # NLTK word rokenize 
    def word_tokenize_wrapper(text):
        return word_tokenize(text)

    df['Tokenizing'] = df['Remove_Special_Text'].apply(word_tokenize_wrapper)

    list_stopwords = stopwords.words('indonesian', 'english')


    # ---------------------------- manualy add stopword  ------------------------------------
    # append additional stopword
    list_stopwords.extend(['jg', 'josss', "yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                        'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                        'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                        'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                        'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'wkwk', 'wkwkwk', 'mjb', 'tbh',
                        'knl', 'teruz', 'zeh', 'tra', 'kks', 'hhahahahahahahahahahahahahahahahahahahahaha',
                        'buahahahahahahahahahahahahahahahahahahahahahahahah', 'mlh', 'pye', 'tauu', 'bjo',
                        'wkwksjsjsksksjjss', 'jis', 'ckckckckckk', 'hrus', 'sgera', 'adl', 'zzz', 'yagaksi',
                        'lg', 'udh', 'dr', 'duarr', 'gw', 's', 'cm'
                        ])

    list_stopwords.extend(["stopwords"][0].split(' '))

    list_stopwords = set(list_stopwords)

    def stopwords_removal(words):
        return [word for word in words if word not in list_stopwords]

    df['Stopword'] = df['Tokenizing'].apply(stopwords_removal) 

    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # stemmed
    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}

    for document in df['Stopword']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '
                
    print(len(term_dict))
    print("------------------------")

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        print(term,":" ,term_dict[term])
        
    print(term_dict)
    print("------------------------")

    # apply stemmed term to dataframe
    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    df['Stemming'] = df['Stopword'].swifter.apply(get_stemmed_term)
    
    df['Dataclean'] = df['Stemming'].apply(lambda x: ' '.join(x))
    
    df.to_csv('uploads/modeltest.csv', index=False)
    
    df = pd.read_csv("uploads/Dataclean.csv", sep=',', encoding='latin1')

    # extract the tweet text and the labels
    text = df["Tweet"].tolist()
    labels = df["Sentiment"].tolist()
    
    # create the transform
    vectorizer = TfidfVectorizer()

    # tokenize and build vocab
    vectorizer.fit(text)

    # encode document
    X = vectorizer.transform(text)
    
    # split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=0)
    
    modelrbf = SVC(kernel="rbf")
    modelrbf.fit(X_train, y_train)
    y_predrbf = modelrbf.predict(X_test)
    
    # load the new data into a DataFrame
    df_new = pd.read_csv("uploads/modeltest.csv", sep=',', encoding='latin1')

    # extract the tweet text
    text_new = df_new["Tweet"].tolist()

    # encode the new data
    X_new = vectorizer.transform(text_new)

    # make predictions using the trained SVM
    predictions = modelrbf.predict(X_new)

    print(predictions)

    df_new["Prediction"] = predictions
    
    df_new = pd.DataFrame(df_new[['Datetime', 'Tweet', 'Prediction']])
    
    df_new.to_csv('uploads/predictedrbf.csv', index=False, encoding='latin1')
    
    data5 = []
    
    with open ('uploads/predictedrbf.csv', encoding='latin1') as f:
        reader = csv.DictReader(f)
        
        [data5.append(dict(row)) for row in reader]
    
    return render_template('prediksisvm.html', data=data5, list=list, len=len)

if __name__ == '__main__':
    app.run(debug=True)