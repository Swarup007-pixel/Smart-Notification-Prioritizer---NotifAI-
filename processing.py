# this file contains all the processing functions that are used to process the data before it is sent to the model for prediction. 
# This includes functions for cleaning, transforming, and feature engineering the data.


#for model1

import emoji
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import string
from nltk .corpus import stopwords
from nltk.stem.porter import PorterStemmer

#weekdays****************************************************************************************************************************
def weekday_and_weekend(day):
    if day in ['Saturday','Sunday']:
        return 0
    else:
        return 1
    

#emoji*******************************************************************************************************************************
def demojizeing_notification(text):
    return emoji.demojize(text)

#model1.loc[:, 'notification_text'] = model1['notification_text'].apply(demojizeing_notification)   # here model 1 is the dataframe.


# nlp preprocessing*******************************************************************************************************************
def NLP(text):
    y = []
    text = text.lower()
    text = nltk.word_tokenize(text)

    for i in text:
        if i .isalnum():
            y.append(i)


    text = y[:]
    y.clear()   

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)  

    text = y[:]
    y.clear()

    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))  

    return " ".join(y)

#model1['notification_text'] = model1['notification_text'].apply(NLP)  # here model 1 is the dataframe.


