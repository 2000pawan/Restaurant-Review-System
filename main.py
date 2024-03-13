# Import Important Library.
import joblib
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re


# Load Model & CountVectorizer using jonlib
  
model=joblib.load('model.joblib')
cv=joblib.load('cv.joblib')

# Import Image Using Pillow Modoule.
image=Image.open('img.jpg')

# Streamlit Function For Building Button and app.
def main():
    st.image(image,width=900)
    st.title('Restaurant Reviews Analysis')
    html_temp='''
    <div style='background-color:red; padding:16px'>
    <h1 style='color:  #000000; text-align: center;'>Resturant Review Machine Learning Model</h1>
    </div>
    <h2 style='color:  red; text-align: center;'>Please Give Valueable Feedback</h2>
    '''
    st.markdown(html_temp,unsafe_allow_html=True)
    corpus=st.text_input('Enter Your Review')
    result=''
    if st.button('Predict',''):
        result=prediction(corpus)
    temp='''
     <div style='background-color:black; padding:12px'>
     <h1 style='color:  red ; text-align: center;'>{}</h1>
     </div>
     '''.format(result)
    st.markdown(temp,unsafe_allow_html=True)
    
# Prediction Function to predict from model.
spwords=list(ENGLISH_STOP_WORDS)
spwords.remove('not')
spwords.remove('no')  
def cleaning_dataset(doc):
    doc=doc.lower()
    doc=re.sub('[^a-z ]','',doc)
    doc=doc.split()
    new_doc=''
    for word in doc:
        if word not in spwords:
            new_doc=new_doc+word+' '
    return new_doc.strip() 
def prediction(corpus):
    corpus_new=list(map(cleaning_dataset,corpus))
    X_test=cv.transform(corpus_new).toarray()
    prediction=model.predict(X_test)
    if prediction==0:
        return 'You Do Not Liked This Restaurant. Thanks For Visit🙏🏼🙏🏼🙏🏼'
    else:
        return 'You Liked This Restaurant. Thanks For Visit🙏🏼🙏🏼🙏🏼. Come Again'

if __name__=='__main__':
    main()


