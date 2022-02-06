#!/usr/bin/env python
# coding: utf-8

# # Language Detection Project

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1-Importing data that we scrapped from twitter

# In[2]:


import pandas as pd
pd.options.mode.chained_assignment = None  
df = pd.read_csv('language_detection_data.csv',index_col=None)
df


# ## 2-Exploring our dataset

# In[3]:


#shape
df.shape


# In[4]:


#different languages
languages = set(df['Language'])
print('Languages', languages)


# In[5]:


#check missing values
df.isnull().sum()


# In[6]:


#display the number of texts available  of every class (language)
df['Language'].value_counts()


# In[9]:


#we can see that our data is balanced
plt.figure(figsize = (10, 8))
sns.countplot(x=df['Language'])
plt.show()


# Note: We see that our dataset is almost balanced

# ## 3- Data Pre-Processing

# In[7]:


#Text Cleaning library
import neattext.functions as nfx
#text cleaning fct
def Clean_Text(data,column):
     #convert text to lower
    data[column]=data[column].str.lower()
    #replace \n and s with space
    data[column].replace(r'\s+|\\n', ' ',regex=True, inplace=True) 
    #remove userhandles
    data[column]=data[column].apply(nfx.remove_userhandles)
    #remove urls
    data[column]=data[column].apply(nfx.remove_urls)
    #remove punctuations
    data[column]=data[column].apply(nfx.remove_punctuations)
    #remove special characters
    data[column]=data[column].apply(nfx.remove_special_characters)
    #remove emails
    data[column]=data[column].apply(nfx.remove_emails)
    #remove multiple space
    data[column]=data[column].apply(nfx.remove_multiple_spaces)
    #replace dates 1-2digits Mon 4digits
    data[column].replace(r'\d{1,2}\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|janv|juil|aot|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|January|February|March|April|May|June|July|August|September|October|November|December|avr|déc|févr|janv|juill|nov|oct|sept)\s\d{4}', ' ',regex=True, inplace=True) 
    data[column].replace("(janv|\dh| h | \d |\d | \d|http|https|a35crasherait| d24d1minfriendly| \d+ \d+| \d+\d+)", "", regex=True, inplace=True)
    data[column].replace("  ", " ",regex=True, inplace=True)
    data[column].replace(r'(autres personnes|en rponse|rponse|en|[a-z][0-9][0-9][a-z]+|[0-9][0-9]+|[0,1,4,6,8]+|[0,1,4,6,8]+|[a-z][0,1,4,6,8])', ' ', regex=True, inplace=True)
    data[column].replace(r'avren|decn|fevren|janven|juilen|noven|octen|septen|avr|déc|févr|janv|juil|nov|oct|sept', ' ', regex=True, inplace=True)
    #replace / 
    data[column].replace('\/', ' ',regex=True, inplace=True)
    #replace '
    data[column].replace('\'', ' ', regex=True, inplace=True)
    return data


# In[8]:


dataset=Clean_Text(df,'Text')
dataset


# In[9]:


#Remove english and french stop words
import nltk
stopwords = set(nltk.corpus.stopwords.words('english')) | set(nltk.corpus.stopwords.words('french'))
dataset['Text'] = dataset['Text'].str.lower().apply(lambda x: ' '.join([word for word in str(x).split() if word not in stopwords]))


# In[10]:


#delete empty rows
dataset = dataset[dataset['Text']!= '']
#reset data index
dataset=dataset.reset_index().drop('index',axis=1)
dataset


# extract_keywords function was created In order to determine the manipulating voccabulary of each language

# In[11]:


from collections import Counter
def extract_keywords(text,num=50):
    tokens=[tok for tok in text.split()]
    most_commen_tokens=Counter(tokens).most_common(num)
    return dict(most_commen_tokens)


# ### i- Dominant voccabulary in Darija

# In[12]:


#extraction of Darija keywords
language_list=dataset['Language'].unique().tolist()
Darija_list=dataset[dataset['Language']=='Darija']['Text'].tolist()
Darija_docx=' '.join(Darija_list)
keywords_Darija=extract_keywords(Darija_docx)
keywords_Darija


# In[16]:


#Word Cloud
from wordcloud import WordCloud
def plot_wordcloud(docx):
    mywordcloud=WordCloud().generate(docx)
    plt.figure(figsize=(20,10))
    plt.imshow(mywordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.show()
plot_wordcloud(Darija_docx) 


# To perform a deeper data cleanning, we tokenize our text into a set of separated words

# In[13]:


import nltk
dataset['tokenized_sents'] = dataset.apply(lambda row: nltk.word_tokenize(row['Text']), axis=1)
dataset


# In[14]:


#remove words with less that 3 letters
def cleaner(dataset):
    for sentence in dataset.tokenized_sents:
        for token in sentence:
            if len(token) < 3  :
                sentence.remove(token)
    return dataset
dataset=cleaner(dataset)
dataset


# In[15]:


#After successfully removing noise from our tokenze we detokenize the sentences
from nltk.tokenize.treebank import TreebankWordDetokenizer
dataset['detokenized_sents'] = dataset.apply(lambda row: TreebankWordDetokenizer().detokenize(row['tokenized_sents']), axis=1)
dataset=dataset[dataset['detokenized_sents'].str.len()>=4]
dataset


# In[16]:


dataset[['detokenized_sents','Language']]


# In[19]:


#Word Cloud
from wordcloud import WordCloud
def plot_wordcloud(docx):
    mywordcloud=WordCloud().generate(docx)
    plt.figure(figsize=(20,10))
    plt.imshow(mywordcloud,interpolation='bilinear')
    plt.axis('off')
    
    
    plt.show()   


# In[20]:


#We check again the voccabulary of each language the results this time are satisfying
#liste of Darija keywords
language_list=dataset['Language'].unique().tolist()
Darija_list=dataset[dataset['Language']=='Darija']['detokenized_sents'].tolist()
Darija_docx=' '.join(Darija_list)
plot_wordcloud(Darija_docx) 


# In[21]:


#liste of English keywords 
language_list=dataset['Language'].unique().tolist()
English_list=dataset[dataset['Language']=='English']['detokenized_sents'].tolist()
English_docx=' '.join(English_list)
plot_wordcloud(English_docx) 


# In[22]:


#Dataset partition (trainning & testing data)
import numpy as np
from sklearn.model_selection import train_test_split

X=dataset['detokenized_sents']
y=dataset['Language']
#we used  80% for training data and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


# ## 3.1- Vectorization

# ### 3.1.1 Uni-gram

# In[23]:


# Extract Unigrams
from sklearn.feature_extraction.text import CountVectorizer
unigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,1))
X_unigram_train_raw = unigramVectorizer.fit_transform(X_train)
X_unigram_test_raw = unigramVectorizer.transform(X_test)

#getFreatures
unigramFeatures = unigramVectorizer.get_feature_names()

print('Number of unigrams in training set:', len(unigramFeatures))


# Our corpus is composed of 32 different features, represented in this list:

# In[25]:


unigramFeatures


# In[24]:


#Distribution of uni-grams through the laguages
def train_lang_dict(X_raw_counts, y_train):
    lang_dict = {}
    for i in range(len(y_train)):
        lang = y_train[i]
        v = np.array(X_raw_counts[i])
        if not lang in lang_dict:
            lang_dict[lang] = v
        else:
            lang_dict[lang] += v
            
    # to relative
    for lang in lang_dict:
        v = lang_dict[lang]
        lang_dict[lang] = v / np.sum(v)
        
    return lang_dict

language_dict_unigram = train_lang_dict(X_unigram_train_raw.toarray(), y_train.values)

# Collect relevant chars per language
def getRelevantCharsPerLanguage(features, language_dict, significance=1e-4):
    relevantCharsPerLanguage = {}
    for lang in languages:
        chars = []
        relevantCharsPerLanguage[lang] = chars
        v = language_dict[lang]
        for i in range(len(v)):
            if v[i] > significance:
                chars.append(features[i])
    return relevantCharsPerLanguage


# In[27]:


language_dict_unigram


# The 32 features are distributed among the languages as follows:

# In[25]:


relevantCharsPerLanguage = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram)
# Print number of unigrams per language
for lang in languages:    
    print(lang, len(relevantCharsPerLanguage[lang]))


# In[26]:


# get most common chars for a few European languages
europeanLanguages = [ 'English', 'French', 'Darija']
relevantChars_OnePercent = getRelevantCharsPerLanguage(unigramFeatures, language_dict_unigram, 1e-2)

# collect and sort chars
europeanCharacters = []
for lang in europeanLanguages:
    europeanCharacters += relevantChars_OnePercent[lang]
europeanCharacters = list(set(europeanCharacters))
europeanCharacters.sort()

# build data
indices = [unigramFeatures.index(f) for f in europeanCharacters]
data = []
for lang in europeanLanguages:
    data.append(language_dict_unigram[lang][indices])

#build dataframe
df = pd.DataFrame(np.array(data).T, columns=europeanLanguages, index=europeanCharacters)
df.index.name = 'Characters'
df.columns.name = 'Languages'

# plot heatmap
import seaborn as sn
import matplotlib.pyplot as plt
sn.set(font_scale=0.8) # for label size
sn.set(rc={'figure.figsize':(10, 10)})
sn.heatmap(df, cmap="Greens", annot=True, annot_kws={"size": 12}, fmt='.0%')# font size
plt.show()


# **Note: in our case using uni-gram won't be usefull since all languages use almost the same letters**

# ### 3.1.2 Bi-gram

# In[27]:


# number of bigrams
from sklearn.feature_extraction.text import CountVectorizer
bigramVectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2))
X_bigram_raw = bigramVectorizer.fit_transform(X_train)
bigramFeatures = bigramVectorizer.get_feature_names()
print('Number of bigrams', len(bigramFeatures))


# Bi-gram helped us to increase the number of features used in our corpus from 32 in uni-gram to 941 features listed bellow

# Note: To determine the significant combinations for each language, the notion of signiphicance is introduced as the minimum weight associated with the representative feature in a language in our case significance is equal to 0.01

# In[31]:


# top bigrams (>1%) for each language
language_dict_bigram = train_lang_dict(X_bigram_raw.toarray(), y_train.values)
relevantCharsPerLanguage = getRelevantCharsPerLanguage(bigramFeatures, language_dict_bigram, significance=1e-2)
print('Darija', relevantCharsPerLanguage['Darija'])
print('French', relevantCharsPerLanguage['French'])
print('English', relevantCharsPerLanguage['English'])


# Note: we can see even if it's better that unigrams but we always have some features that are the same in the 3 languages. However our objective is to find features that makes each language (unique for this language) so we can distiguich between them. Alternatively, we could also use a mixture of Uni-Grams and Bi-Grams, restricted on the most frequently used ones.

# ### # **3.1.1 Mixture Uni-gram & Bi-gram using top 1% features**

# In[1]:


#When we restrict ourselves to a limited number of features, it is important, that we will capture details for each language.


# In[28]:


# Uni- & Bi-Gram Mixture CountVectorizer for top 1% features
from sklearn.feature_extraction.text import CountVectorizer

top1PrecentMixtureVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2), min_df=1e-2)
X_top1Percent_train_raw = top1PrecentMixtureVectorizer.fit_transform(X_train)
X_top1Percent_test_raw = top1PrecentMixtureVectorizer.transform(X_test)

language_dict_top1Percent = train_lang_dict(X_top1Percent_train_raw.toarray(), y_train.values)

top1PercentFeatures = top1PrecentMixtureVectorizer.get_feature_names()
print('Length of features', len(top1PercentFeatures))
print('')

#Unique features per language
relevantChars_Top1Percent = getRelevantCharsPerLanguage(top1PercentFeatures, language_dict_top1Percent, 1e-5)
for lang in relevantChars_Top1Percent:
    print("{}: {}".format(lang, len(relevantChars_Top1Percent[lang])))


# In[79]:


dataset['Text'][12000]


# In[80]:


test3 = top1PrecentMixtureVectorizer.transform([dataset['Text'][12000]])
var_test3=toNumpyArray(test3)
new_var= np.array(test3.toarray()[:,relevantColumnIndices])
new_var


# Well, we can restrict ourselves to the top 60 Uni- & Bi-Grams per language.This will lead to max 3 * 60 = 180 features

# In[29]:


def getRelevantGramsPerLanguage(features, language_dict, top=60):
    relevantGramsPerLanguage = {}
    for lang in languages:
        chars = []
        relevantGramsPerLanguage[lang] = chars
        v = language_dict[lang]
        sortIndex = (-v).argsort()[:top]
        for i in range(len(sortIndex)):
            chars.append(features[sortIndex[i]])
    return relevantGramsPerLanguage

top60PerLanguage_dict = getRelevantGramsPerLanguage(top1PercentFeatures, language_dict_top1Percent)

# top60
allTop60 = []
for lang in top60PerLanguage_dict:
    allTop60 += set(top60PerLanguage_dict[lang])

top60 = list(set(allTop60))
    
print('All items:', len(allTop60))
print('Unique items:', len(top60))


# And here'are the top 60 used features in our dataset

# In[78]:



print('Darija', top60PerLanguage_dict['Darija'])
print('French', top60PerLanguage_dict['French'])
print('English',top60PerLanguage_dict['English'])


# In[ ]:


#So, when dealing with the top-60-approach on these 3 languages, we will effectively use 92 features only.
#Conclusion: From a theoretical perspective, it is most efficient to use a Mixture of the most common Uni-Grams and Bi-Grams.


# Now, let's build the data set for the models, based on our 92 features

# In[30]:


# getRelevantColumnIndices
def getRelevantColumnIndices(allFeatures, selectedFeatures):
    relevantColumns = []
    for feature in selectedFeatures:
        relevantColumns = np.append(relevantColumns, np.where(allFeatures==feature))
    return relevantColumns.astype(int)

relevantColumnIndices = getRelevantColumnIndices(np.array(top1PercentFeatures), top60)


X_top60_train_raw = np.array(X_top1Percent_train_raw.toarray()[:,relevantColumnIndices])
X_top60_test_raw = X_top1Percent_test_raw.toarray()[:,relevantColumnIndices] 

print('train shape', X_top60_train_raw.shape)
print('test shape', X_top60_test_raw.shape)


# ## 4- Modeling

# ## Naive Bayes

# Now, we can apply the Naive Bayes on our different feature sets: .Unigram (X_unigram_train_raw) .Mixture Top 1% (X_top1Percent_train_raw) .Mixture Top 60 (X_top60_train_raw)

# In[31]:


# Define some functions for our purpose

from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import matplotlib.pyplot as plt
import scipy

# Utils for conversion of different sources into numpy array
def toNumpyArray(data):
    data_type = type(data)
    if data_type == np.ndarray:
        return data
    elif data_type == list:
        return np.array(data_type)
    elif data_type == scipy.sparse.csr.csr_matrix:
        return data.toarray()
    print(data_type)
    return None
def normalizeData(train, test):
    train_result = normalize(train, norm='l2', axis=1, copy=True, return_norm=False)
    test_result = normalize(test, norm='l2', axis=1, copy=True, return_norm=False)
    return train_result, test_result

def applyNaiveBayes(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict,clf

def plot_F_Scores(y_test, y_predict):
    f1_micro = f1_score(y_test, y_predict, average='micro')
    f1_macro = f1_score(y_test, y_predict, average='macro')
    f1_weighted = f1_score(y_test, y_predict, average='weighted')
    print("F1: {} (micro), {} (macro), {} (weighted)".format(f1_micro, f1_macro, f1_weighted))

def plot_Confusion_Matrix(y_test, y_predict, color="Blues"):
    allLabels = list(set(list(y_test) + list(y_predict)))
    allLabels.sort()
    confusionMatrix = confusion_matrix(y_test, y_predict, labels=allLabels)
    unqiueLabel = np.unique(allLabels)
    df_cm = pd.DataFrame(confusionMatrix, columns=unqiueLabel, index=unqiueLabel)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    
    sn.set(font_scale=0.8) # for label size
    sn.set(rc={'figure.figsize':(15, 15)})
    sn.heatmap(df_cm, cmap=color, annot=True, annot_kws={"size": 12}, fmt='g')# font size
    plt.show()


# ### Uni-grams

# In[32]:


# Unigrams
X_unigram_train, X_unigram_test = normalizeData(X_unigram_train_raw, X_unigram_test_raw)
y_predict_nb_unigram,clf1 = applyNaiveBayes(X_unigram_train, y_train, X_unigram_test)
plot_F_Scores(y_test, y_predict_nb_unigram)
plot_Confusion_Matrix(y_test, y_predict_nb_unigram, "Oranges")


# ### Top 1% Mixture

# In[33]:


# Top 1%
X_top1Percent_train, X_top1Percent_test = normalizeData(X_top1Percent_train_raw, X_top1Percent_test_raw)
y_predict_nb_top1Percent,clf2 = applyNaiveBayes(X_top1Percent_train, y_train, X_top1Percent_test)
plot_F_Scores(y_test, y_predict_nb_top1Percent)
plot_Confusion_Matrix(y_test, y_predict_nb_top1Percent, "Reds")


# ### Top 60 Mixture

# In[34]:


# Top 60
X_top60_train, X_top60_test = normalizeData(X_top60_train_raw, X_top60_test_raw)
y_predict_nb_top60,clf3 = applyNaiveBayes(X_top60_train, y_train, X_top60_test)
plot_F_Scores(y_test, y_predict_nb_top60)
plot_Confusion_Matrix(y_test, y_predict_nb_top60, "Greens")


# # Comparison
# **For Naive Bayes we achieve the following scores for F1 (weighted):
# 
# Unigram: 0.8387990185718953 (using 32 features)
# 
# Top 1% Mixture: 0.9176212136387736 (471 features)
# 
# Top 60 Mixture: 0.8889191792260633 (92 features)**
# 
# we notice that top 1% Mixture represent languages the most comparing with unigrams and top 60.

# In[36]:


from sklearn.metrics import classification_report


# In[37]:


print('Unigram')
print()
print(classification_report( y_predict_nb_unigram , y_test))


# In[38]:


print('Top 1% Mixture')
print()
print(classification_report(  y_predict_nb_top1Percent , y_test))


# In[39]:


print('Top 60 Mixture')
print()
print(classification_report(  y_predict_nb_top60 , y_test))


# ## K Nearest Neighbor

# ### Unigrams

# The default for k=5. Let's stick to that default

# In[40]:


from sklearn.neighbors import KNeighborsClassifier

def applyNearestNeighbour(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = KNeighborsClassifier()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict,clf

#Unigrams
y_predict_knn_unigram,clf4 = applyNearestNeighbour(X_unigram_train, y_train, X_unigram_test)
plot_F_Scores(y_test, y_predict_knn_unigram)
plot_Confusion_Matrix(y_test, y_predict_knn_unigram, "Purples")


# ### choose the best value of K

# In[41]:


#Choose the right value of k suitable, for this we will create a loop that trains various KNN models with different
#values of k, then calculate the error rate for each of these models and put them in a list in order to compare them
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_unigram_train, y_train)
    pred_i = knn.predict( X_unigram_test)
    error_rate.append(np.mean(pred_i != y_test))
 

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title("Taux d'erreurs vs. valeurs de K ")
plt.xlabel('K')
plt.ylabel("Taux d'erreurs")


# we can choose 12 as the best value of K

# In[42]:


#Uni-grams
from sklearn.neighbors import KNeighborsClassifier

def applyNearestNeighbour(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = KNeighborsClassifier(n_neighbors=12)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict,clf

#Unigrams
y_predict_knn_unigram,clf4 = applyNearestNeighbour(X_unigram_train, y_train, X_unigram_test)
plot_F_Scores(y_test, y_predict_knn_unigram)
plot_Confusion_Matrix(y_test, y_predict_knn_unigram, "Purples")


# ### Top 1% Mixture

# In[43]:


# Top 1%
y_predict_knn_top1P,clf5 = applyNearestNeighbour(X_top1Percent_train, y_train,X_top1Percent_test)
plot_F_Scores(y_test, y_predict_knn_top1P)
plot_Confusion_Matrix(y_test, y_predict_knn_top1P, "Blues")


# ### Top 60 Mixture

# In[44]:


# Top 60
y_predict_knn_top60,clf6 = applyNearestNeighbour(X_top60_train, y_train, X_top60_test)
plot_F_Scores(y_test, y_predict_knn_top60)
plot_Confusion_Matrix(y_test, y_predict_knn_top60, "Blues")


# # Comparison
# **For KNN we achieve the following scores for F1 (weighted):
# 
# Unigram: 0.8470151344064695 (using 32 features)
# 
# Top 1% Mixture: 0.8952434847125752 (471 features)
# 
# Top 60 Mixture: 0.888294583348349 (92 features)**
# 
# in our case, we can see that that top 1% Mixture represent languages the most.

# In[45]:


print('Unigram')
print()
print(classification_report( y_predict_knn_unigram, y_test))


# In[46]:


print('Top 1% Mixture')
print()
print(classification_report( y_predict_knn_top1P , y_test))


# In[47]:


print('Top 60 Mixture')
print()
print(classification_report( y_predict_knn_top60, y_test))


# ## Logistic Regression

# ### Uni-grams

# In[48]:


from sklearn.linear_model import LogisticRegression

def applyLogisticRegression(X_train, y_train, X_test):
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict,clf

## Unigrams
y_predict_RL_unigram,clf7 = applyLogisticRegression(X_unigram_train, y_train, X_unigram_test)
plot_F_Scores(y_test, y_predict_RL_unigram)
plot_Confusion_Matrix(y_test, y_predict_RL_unigram, "Purples")


# ### Top 1% Mixture

# In[49]:


# Top 1%
y_predict_RL_top1P,clf8 = applyLogisticRegression(X_top1Percent_train, y_train,X_top1Percent_test)
plot_F_Scores(y_test, y_predict_RL_top1P)
plot_Confusion_Matrix(y_test, y_predict_RL_top1P, "Blues")


# ### Top 60 Mixture

# In[50]:


# Top 60
y_predict_RL_top60,clf9 = applyLogisticRegression(X_top60_train, y_train, X_top60_test)
plot_F_Scores(y_test, y_predict_RL_top60)
plot_Confusion_Matrix(y_test, y_predict_RL_top60, "Blues")


# # Comparison
# **For Logistic Regression we achieve the following scores for F1 (weighted):
# 
# Unigram: 0.8685706731912578 (using 32 features)
# 
# Top 1% Mixture: 0.9394186585277219 (471 features)
# 
# Top 60 Mixture: 0.9269944856641448 (92 features)**
# 
# in our case, we can see that that top 1% Mixture represent languages the most.

# In[51]:


print('Unigram')
print()
print(classification_report( y_predict_RL_unigram, y_test))


# In[52]:


print('Top 1% Mixture')
print()
print(classification_report( y_predict_RL_top1P, y_test))


# In[53]:


print('Top 60 Mixture')
print()
print(classification_report( y_predict_RL_top60, y_test))


# **To improve the top 1% and have better result we can use Tunning parameters:**

# In[49]:


#Tunning param
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV 
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=clf8, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_top1Percent_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# we can see that the score of prediction has improved

# ## 5-Evaluating

# **To choose the final model we applied k-Fold Cross Validation**

# In[50]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#k-fold cross validation fct
def ten_fold_cross(model,X_train,y_train):
            cv = KFold(n_splits=10, random_state=1, shuffle=True)
            # create model
            # evaluate model
            scores = cross_val_score(model,  X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
            # report performance
            print(scores)
            print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
            print()
            return np.mean(scores)

#fct to make sure  that the model is well generalized
from sklearn.metrics import accuracy_score
def compare_accuracy_after_and_before_cross(y_test,y_predict,scores):
                    accuracy=accuracy_score(y_test, y_predict)
                    print('before cross validation, accuracy= ',accuracy)
                    print()
                    print('after cross validation, accuracy= ',scores)
                    
                    


# In[51]:


#Unigrams NB model
scores=ten_fold_cross(clf1,X_unigram_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_nb_unigram,scores)


# we can see that there is no big difference between accuracy before and after cross validation, so our model is not overfitting

# In[52]:


#Top 1% NB Model
scores1=ten_fold_cross(clf2,X_top1Percent_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_nb_top1Percent,scores1) 


# In[53]:


#Top 60 NB Model
scores2=ten_fold_cross(clf3,X_top60_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_nb_top60,scores2) 


# In[54]:


#Uni-grams KNN  model
scores3=ten_fold_cross(clf4,X_unigram_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_knn_unigram,scores3)


# In[55]:


#Top 1% KNN Model
scores4=ten_fold_cross(clf5,X_top1Percent_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_knn_top1P,scores4) 


# In[56]:


#Top 60 KNN Model
scores5=ten_fold_cross(clf6,X_top60_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_knn_top60,scores5) 


# In[57]:


#Uni-grams Logistic Regression  model
scores6=ten_fold_cross(clf7,X_unigram_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_RL_unigram,scores6)


# In[58]:


#Top 1% Logistic Regression Model
scores7=ten_fold_cross(clf8,X_top1Percent_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_RL_top1P,scores7) 


# In[59]:


#Top 60 Logistic Regression Model
scores8=ten_fold_cross(clf9,X_top60_train, y_train)
compare_accuracy_after_and_before_cross(y_test, y_predict_RL_top60,scores8) 


# # Comparaison
# **after applying cross validation we found: Uni-grams NB model: 0.8318961319872631
# 
# Top 1% NB model: 0.9112799084810174
# 
# Top 60 NB model: 0.8827421655183706
# 
# Uni-grams KNN model 0.8330392575070127
# 
# Top 1% KNN model: 0.8853383272363766
# 
# Top 60 KNN model: 0.8768289792875825
# 
# Uni-grams LR model 0.8688391652986216
# 
# Top 1% LR model: 0.9355602306071535
# 
# Top 60 LR model: 0.9155337745547929
# 
# we can see that the best model in our case is Logistic Regression using the Top 1% Mixture. we'll use it in the next section to predict new data.**

# ## Predicting with some more data

# In[60]:


#create a function  that takes a text as input and return the sitable laguage 
def detect_language(text):
            #vectorize the text
            test = top1PrecentMixtureVectorizer.transform([text])
            var_test=toNumpyArray(test)
            l=clf8.predict(var_test)
            #Check for the prediction probability
            pred_proba=clf8.predict_proba(var_test)
            pred_percentage_for_all=dict(zip(clf8.classes_,pred_proba[0]))
            print("Prediction using Logistic Regression Top 1%:  : {} , Prediction Score : {}".format(l[0],np.max(pred_proba)))
            print()
            print(pred_percentage_for_all)


# In[61]:


#test text in darija
detect_language('la walakin im not sure that she would be hya')


# In[62]:


#test text in english
detect_language('hello world im so happy today')


# In[63]:


#test text in french
detect_language('je suis tres heureuse aujourd hui, je me sens tres bien')


# # Error Analysis

# In[64]:


def plotTopErrors(y_predict, top=5):
    ys = y_test.values
    Xs = X_test.values
    errorCount = 0
    
    for i in range(len(ys)):
        if not ys[i]==y_predict[i]:
            errorCount += 1
            print("#{}: Expected: {}, Predicted: {}".format(errorCount, ys[i], y_predict[i]))
            print("Text:", Xs[i])
            print("=================================================")
        if errorCount >= top:
            break


# In[65]:


plotTopErrors(y_predict_RL_top1P, top=10)


# # Thank you !
