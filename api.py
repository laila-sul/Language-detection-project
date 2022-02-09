from flask import Flask,request, jsonify, render_template
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
# Uni- & Bi-Gram Mixture CountVectorizer for top 1% features
from sklearn.feature_extraction.text import CountVectorizer
import scipy



app=Flask(__name__)
model=pickle.load(open('language_prediction.pickle','rb'))
vectorizer=pickle.load(open('countvectorizer.pickle','rb'))


# @app.route('/')
# def home():
         
#          return render_template("index.html")

@app.route('/',methods=['GET'])
def predict():
    
    if request.method=="GET":
                text=request.args.get('Text')
                if text is None:
                   return render_template("index.html",lan='',text='')
                else:
                    #vectorize the text
                    test = vectorizer.transform([str(text)])
                    
                    #var_test=toNumpyArray(test)
                    l=model.predict(test.toarray())
                    #Check for the prediction probability
                    #pred_proba=model.predict_proba(test.toarray())
                    #pred_percentage_for_all=dict(zip(model.classes_,pred_proba[0]))
                    #output=round(l[0])
                    
                    return render_template("index.html",lan=l[0],text=text)
    #return l[0]
    #jsonify({'the language is':l[0]})
    


if __name__=="__main__":
           app.run(debug=True)