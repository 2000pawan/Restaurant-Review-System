from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
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