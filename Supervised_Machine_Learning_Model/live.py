import pandas as pd 
from bokeh.plotting import figure, show
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer, word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

# nltk.download('punkt_tab') # Downlaaded Had to download these two
# nltk.download('wordnet') # Downloaded

#NOTE Great Resource for building text classifier.
# It's what I used to write this program
# https://towardsdatascience.com/step-by-step-basics-text-classifier-e666c6bac52b/

df = pd.read_csv('spam.csv', usecols= [0, 1], names= ['class', 'text'], header= 0) # To read the dataset
print(df.head())

print('\nNumber of rows initially:', df.shape[0])

# Clean
df1 = df.dropna()
print('\nNum of rows after dropna:', df1.shape[0])

df2 = df1.drop_duplicates()
print('\nNum of rows after drop duplicates:', df2.shape[0])

# Clean the data to be safe
df2['class'] = df2['class'].str.strip().str.lower()
        
counts = df2['class'].value_counts() # No more need for the for loop
print(counts)

fig = plt.figure()
ax = fig.add_subplot()
ax.bar(x= counts.index, height= counts.values)
ax.set_title('ham vs spam emails')
ax.set_xlabel('class')
ax.set_ylabel('frequency')
# plt.show()


# Cleaning, removing punctuation
#  exploring patterns in the text to assess how best to cleanse the data
punc_list = ['.', ',', '!', '#', ';', '�', '-', '@'] # list of special characters/punctuation to search for in data
                                                      # ?'s are a no-no
    # Find puctuation and characters
def punc_search(df, col, pat):
    """
    function that counts the number of narratives
    that contain a pre-defined list of special
    characters and punctuation
    """
    for p in pat:
        v = df[col].str.contains(p).sum() # total n_rows that contain the pattern
        print(f'{p} special character is present in {v} entries')

punc_search(df2, 'text', punc_list)

    # Remove punctuation and charactors
lemmatizer = WordNetLemmatizer()  # initiating lemmatiser object

def text_cleanse(df, col):
    """
    cleanses text by removing special
    characters and lemmatizing each
    word
    """
    df[col] = df[col].str.lower()  # convert text to lowercase
    df[col] = df[col].str.replace(r'.','') # replace punctualion and charactors
    df[col] = df[col].str.replace(r',','') 
    df[col] = df[col].str.replace(r'!','') 
    df[col] = df[col].str.replace(r'#','')  
    df[col] = df[col].str.replace(r';','')  
    df[col] = df[col].str.replace(r'�','')  
    df[col] = df[col].str.replace(r'-','') 
    df[col] = df[col].str.replace(r'@','')  
    df[col] = df[col].str.replace(r';+[a-zA-Z]s+',' ') # remove single characters
    df[col] = df.apply(lambda x: word_tokenize(x[col]), axis=1) # tokenise text ready for lemmatisation
    df[col] = df[col].apply(lambda x:[lemmatizer.lemmatize(word, 'v') for word in x]) # lemmatise words, use 'v' argument to lemmatise versbs (e.g. turns past participle of a verb to present tense)
    df[col] = df[col].apply(lambda x : " ".join(x)) # de-tokenise text ready for vectorisation
    return df
    
df_kleen = text_cleanse(df2, 'text')
print(df2.head())
print(df_kleen.head(15))

# Building the model
    # Split test-train
X_train, X_test,  y_train, y_test = train_test_split(df_kleen['text'], # Features
                                                     df_kleen['class'], # Target
                                                     test_size= 30, # 30% test 70% train
                                                     random_state= 42,
                                                     shuffle= True, # shuffle data before it's split
                                                     stratify= df_kleen['class'] 
                                                     )
    # Text Vectorization
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

    # Model Selection
models = [RandomForestClassifier(n_estimators = 100, max_depth=5, random_state=42), 
          LinearSVC(random_state=42),
          MultinomialNB(), 
          LogisticRegression(random_state=42)]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1) # With StratifiedKFold, the folds are made by preserving the percentage of samples for each class.

scoring = ['accuracy', 'f1_macro', 'recall_macro', 'precision_macro']

#  iterative loop print metrics from each model
for model in models:
    model_name = model.__class__.__name__
    result = cross_validate(model, X_train_vectorized, y_train, cv=kf, scoring=scoring)
    print("%s: Mean Accuracy = %.2f%%; Mean F1-macro = %.2f%%; Mean recall-macro = %.2f%%; Mean precision-macro = %.2f%%" 
          % (model_name, 
             result['test_accuracy'].mean()*100, 
             result['test_f1_macro'].mean()*100, 
             result['test_recall_macro'].mean()*100, 
             result['test_precision_macro'].mean()*100))
    
    # Continue Later. 

