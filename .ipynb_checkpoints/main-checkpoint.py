# Import Important Library.
import pickle
import streamlit as st
from PIL import Image
import package.cleaning as clean

# Load Model & CountVectorizer using Pickle
pickle_in1 = open('model.pkl', 'rb')   
model=pickle.load(pickle_in1)
pickle1 = open('cv.pkl', 'rb') 
cv=pickle.load(pickle1)


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
def prediction(corpus):
    corpus_new=clean.cleaning_dataset(corpus)
    X_test=cv.transform([corpus_new]).toarray()
    prediction=model.predict(X_test)
    if prediction==0:
        return 'You Do Not Liked This Restaurant. Thanks For Visit🙏🏼🙏🏼🙏🏼'
    else:
        return 'You Liked This Restaurant. Thanks For Visit🙏🏼🙏🏼🙏🏼. Come Again'

if __name__=='__main__':
    main()


