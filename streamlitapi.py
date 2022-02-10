import pickle
import streamlit as st

model=pickle.load(open('language_prediction.pickle','rb'))
vectorizer=pickle.load(open('countvectorizer.pickle','rb'))
def main():
    st.title('Language Prediction')
    #input text
    text=st.text_area('Predict Text Language (English,French,Moroccan Dialect Darija): ')
    #prediction code
    if st.button('Detect Text Language'):
        #vectorize the text
        test = vectorizer.transform([str(text)])
        #var_test=toNumpyArray(test)
        l=model.predict(test.toarray())
        #output=round(l[0],2)
        st.success('The predicted Language is: {}'.format(l[0]))
        
if __name__=="__main__":
        main()    