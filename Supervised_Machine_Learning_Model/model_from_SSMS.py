import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from nltk import WordNetLemmatizer, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

# Loading table fro SQL Server database to a Pandas DF
#-----------------------------------------------------------------------------------------------------------------------------------------
# Configure connection details
server_name = 'localhost'
database_name = 'Email_Classify'
trusted_connection = 'yes' # or 'no' if using username and password
# username = 'SesethuMBango'
# password = '******' # No need for username and password
conn_string = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server_name};"
    f"DATABASE={database_name};"
    f"Trusted_Connection={trusted_connection}"
)
try:
    cnxn = pyodbc.connect(conn_string)
    cursor = cnxn.cursor()
    print('Connection to SQL server successfull')
except pyodbc.Error as ex:
    sqlstate = ex.args[0]
    print(f'Error connecting to SQL Server: {sqlstate}')
    exit()

# Load data from SQL database table onto a Pandas dataframe
df = pd.read_sql_query('SELECT class, text FROM Spam', cnxn)

cursor.close()
cnxn.close()

print(df.head())
#---------------------------------------------------------------------------------------------------------

# Clean
df2 = df.dropna()
print('\nNum of rows after dropna:', df2.shape[0])

# df2 = df1.drop_duplicates()
# print('\nNum of rows after drop duplicates:', df2.shape[0])

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
plt.show()


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
                                                     test_size= 2, # 2% test 98% train Because test data is not going to be used at all in the models
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

print('\n')
#  iterative loop print metrics from each model
for model in models:
    model_name = model.__class__.__name__ # Intersting, getting the model name
    result = cross_validate(model, X_train_vectorized, y_train, cv=kf, scoring=scoring) #NOTE cross_validate trains and validates the training just like that, in one line of code.
    print("%s: Mean Accuracy = %.2f%%; Mean F1-macro = %.2f%%; Mean recall-macro = %.2f%%; Mean precision-macro = %.2f%%" 
          % (model_name, 
             result['test_accuracy'].mean()*100, 
             result['test_f1_macro'].mean()*100, 
             result['test_recall_macro'].mean()*100, 
             result['test_precision_macro'].mean()*100))
    
print('\nLinearSVC models performs the best.')
# FIN. Models perform nearly perfectly

