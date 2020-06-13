import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string

stop = stopwords.words('english')
stem = PorterStemmer()
lem = WordNetLemmatizer()

df = pd.read_json('C:/Users/julperez/Downloads/654841_1157617_bundle_archive/is_train.json')
dfoos = pd.read_json('C:/Users/julperez/Downloads/654841_1157617_bundle_archive/oos_train.json')
df.append(dfoos)

df = df.rename(columns={0:'Intent', 1:'Categories'})

def clean(text):
    text = text.lower()
    text = word_tokenize(text)
    noPuntcText = [w for w in text if w not in string.punctuation]
    clean = [w for w in noPuntcText if w not in stop]
    poststem = [stem.stem(w) for w in clean]
    postlem = [lem.lemmatize(w) for w in poststem]
    return ' '.join(postlem)

df['Intent'] = df['Intent'].apply(lambda x: clean(x))

word_count = df['Intent']
lables = df.Categories

X_train, X_test, y_train, y_test = train_test_split(word_count, lables, test_size=0.20,random_state=0,shuffle=True)

text_clf = Pipeline([
    ('count', CountVectorizer()),
    ('tfidf', TfidfTransformer(sublinear_tf=True)),
    ('Mnb', LinearSVC())
])

text_clf.fit(X_train,y_train)
print(text_clf.score(X_test,y_test), ' score Mnb')

examples = ['find my cellphone for me!','for this design, what company did it']
predicton = text_clf.predict(examples)
print(predicton)