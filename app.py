import os
from flask import Flask,render_template,request,redirect,url_for,flash
from werkzeug.utils import secure_filename
import os
import pandas as pd

import numpy as np
from tqdm import tqdm
import re    #file used for  regular expression
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix,plot_confusion_matrix
import copy
import glob

UPLOAD_FOLDER = 'C:/Users/ELECTROBOT\Documents/working folder/Offensive-Language-Detection-master/Offensive_tweets/offensiveflask/uploads'
ALLOWED_EXTENSIONS = {'csv','tsv'}

app= Flask(__name__)
app.secret_key=b'secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',methods=["GET","POST"])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            cda='file:///C:/Users/ELECTROBOT/Documents/working%20folder/Offensive-Language-Detection-master/Offensive_tweets/offensiveflask/uploads/'
            file1="".join([cda,filename])
            print(file1)
                       
            data = pd.read_csv(file1, sep='\t', header=0)  #header is (id,tweet,subtask_A,subtask_B,subtask_C) seperated by Tab
            tweetscolumn = data[["tweet"]]
            subtsk1 = data[["subtask_a"]]    
            subtsk2 = data.query("subtask_a == 'OFF'")[["subtask_b"]]     #query
            subtsk3 = data.query("subtask_b == 'TIN'")[["subtask_c"]]     # query 
            dublicatetweets = copy.deepcopy(tweetscolumn) 
            

            def preprocessingfunction(tweet):
               undesiredwords = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m']
    
               for x in undesiredwords :
                 tweet = tweet.replace(x, '')    
               tweets =re.sub(r'[^a-zA-Z]', ' ', tweet)  
               lower_tweet = tweets.lower()  
    
               tokens= word_tokenize(lower_tweet) 
               clean_tokens = []
               stopWords = set(stopwords.words('english')) 
               for j in tokens:           
                 if j not in stopWords:   
                    if j.replace(' ', '') != '': 
                         if len(j) > 1:         
                             clean_tokens.append(j)
                   
               clean_token = []
               for x in clean_tokens:
                   x = wordnet_lemmatizer.lemmatize(x) 
                   x = lancaster_stemmer.stem(x)   
                   if len(x) > 1:   
                     clean_token.append(x)           
               return clean_token

            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            tqdm.pandas(desc="Cleaning Data Phase I...")
            dublicatetweets['tweet'] =tweetscolumn['tweet'].progress_apply(preprocessingfunction) 
            textlist = dublicatetweets['tweet'].tolist()   
            def tfid(text_vector):
                 vectorizer = TfidfVectorizer()
                 textvect =[' '.join(tweet) for tweet in text_vector] 
                 vectorizer = vectorizer.fit(textvect)  
                 vectors = vectorizer.transform(textvect).toarray() # tranform function is Transform documents to document-term matrix
                 return vectors
            def makepair(vectors, labels, keyword):
                   result = list()  
                   for vector, label in zip(vectors, labels): 
                       if label == keyword:
                           result.append(vector) 
                   return result
            vectors_a = tfid(textlist) # Numerical Vectors A
            labels_a = subtsk1['subtask_a'].values.tolist() # Subtask A Labels

            vectors_b = makepair(vectors_a, labels_a, "OFF") # Numerical Vectors B
            labels_b = subtsk2['subtask_b'].values.tolist() # Subtask B Labels 

            vectors_c = makepair(vectors_b, labels_b, "TIN") # Numerical Vectors C
            labels_c = subtsk3['subtask_c'].values.tolist() # Subtask C Labels              

            def clffunction(vectors, labels, type="MNB", type1="a"):
                train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors, labels)

                classifier = None
                if(type=="MNB"):
                     classifier = MultinomialNB(alpha=0.7) #smoothening parameter is alpha
                     classifier.fit(train_vectors, train_labels)
                if(type1=="a"):     
                     accuracy = accuracy_score(train_labels, classifier.predict(train_vectors))*100   #predict is used to Perform classification on an array of train or test vectors
                     print("Training model's Accuracy :", accuracy)
                     acc1=int(accuracy)
                     arr1=np.array(acc1)
                     pnan,pyan , pnay, pyay  = confusion_matrix(train_labels, classifier.predict(train_vectors)).ravel()
                     arr1 = np.append(arr1, pnan)
                     arr1 = np.append(arr1, pyan)
                     arr1 = np.append(arr1, pnay)
                     arr1 = np.append(arr1, pyay)   
                     test_predictions = classifier.predict(test_vectors)
                     accuracy1 = accuracy_score(test_labels, test_predictions)*100
                     acc2=int(accuracy1)
                     arr1=np.append(arr1,acc2)
                     print("Test model's Accuracy:", accuracy1)
                     print("Confusion Matrix:", )
                     pnan,pyan , pnay, pyay = confusion_matrix(test_labels, test_predictions).ravel()
                     pnan1,pyan1 , pnay1, pyay1 = confusion_matrix(test_labels, test_predictions).ravel()
                     arr1 = np.append(arr1, pnan1)
                     arr1 = np.append(arr1, pyan1)
                     arr1 = np.append(arr1, pnay1)
                     arr1 = np.append(arr1, pyay1)
                     print(arr1)
                     return arr1
                else: 
                    accuracy = accuracy_score(train_labels, classifier.predict(train_vectors))*100   #predict is used to Perform classification on an array of train or test vectors
                    print("Training model's Accuracy :", accuracy)
                    acc1=int(accuracy)
                    arr1=np.array(acc1)
                    piai,pgai,poai,agpi,agpg,agpo,aopi,aopg,aopo  = confusion_matrix(train_labels, classifier.predict(train_vectors)).ravel()
                    arr1 = np.append(arr1, piai)
                    arr1 = np.append(arr1, pgai)
                    arr1 = np.append(arr1, poai)
                    arr1 = np.append(arr1, agpi)
                    arr1 = np.append(arr1, agpg)
                    arr1 = np.append(arr1, agpo)
                    arr1 = np.append(arr1, aopi)
                    arr1 = np.append(arr1, aopg)
                    arr1 = np.append(arr1, aopo)
                    test_predictions = classifier.predict(test_vectors)
                    accuracy1 = accuracy_score(test_labels, test_predictions)*100
                    acc2=int(accuracy1)
                    arr1=np.append(arr1,acc2)
                    print("Test model's Accuracy:", accuracy1)
                    print("Confusion Matrix:", )
                    piai1,pgai1,poai1,agpi1,agpg1,agpo1,aopi1,aopg1,aopo1= confusion_matrix(test_labels, test_predictions).ravel()
                    arr1 = np.append(arr1, piai)
                    arr1 = np.append(arr1, pgai)
                    arr1 = np.append(arr1, poai)
                    arr1 = np.append(arr1, agpi)
                    arr1 = np.append(arr1, agpg)
                    arr1 = np.append(arr1, agpo)
                    arr1 = np.append(arr1, aopi)
                    arr1 = np.append(arr1, aopg)
                    arr1 = np.append(arr1, aopo)
                    print(arr1)
                    return arr1
#file:///C:/Users/ELECTROBOT/Documents/working%20folder/Offensive-Language-Detection-master/Offensive_tweets/offensiveflask/uploads/

            print("\naccuracy to find wheather it is offence or not ")
            ad=np.array(clffunction(vectors_a[:], labels_a[:], "MNB", "a"))
            print("\naccuracy to find wheather it is Targeted offensive or untargeted offenceive ")
            ad=np.append(ad,clffunction(vectors_b[:], labels_b[:], "MNB", "a"))
            print("\naccuracy to find wheather it is targeted to individual,group,others")
            ad=np.append(ad,clffunction(vectors_c[:], labels_c[:], "MNB", "c"))
            pat=r'C:\Users\ELECTROBOT\Documents\working folder\Offensive-Language-Detection-master\Offensive_tweets\offensiveflask\uploads\\'
            os.chmod(pat,0o777)
            
            print(pat+filename)
            os.remove(pat+filename)


         

            return render_template('sec.html',file=filename, oftr=ad[0],pnan=ad[1],pyan=ad[2],aypn=ad[3],aypy=ad[4],ofte=ad[5],pnpn1=ad[6],pyan1=ad[7],aypn1=ad[8],aypy1=ad[9],trtra=ad[10],pnan2=ad[11],pyan2=ad[12],aypn2=ad[13],aypy2=ad[14],trtest=ad[15],pnpn3=ad[16],pyan3=ad[17],aypn3=ad[18],aypy3=ad[19],indtr=ad[20],piai=ad[21],pgai=ad[22],poai=ad[23],agpi=ad[24],agpg=ad[25],agpo=ad[26],aopi=ad[27],aopg=ad[28],aopo=ad[29],aindtest=ad[30],piai1=ad[31],pgai1=ad[32],poai1=ad[33],agpi1=ad[34],agpg1=ad[35],agpo1=ad[36],aopi1=ad[37],aopg1=ad[38],aopo1=ad[39])
    return render_template('index.html')


if __name__=="__main__" :
    app.run(debug=True,port=8000)   