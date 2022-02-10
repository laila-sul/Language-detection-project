# Language-detection-project
Preforming language detection on several texts using machine learning algorithms.

methodology followed in the project is represented in this map:


<img width="413" alt="image" src="https://user-images.githubusercontent.com/81523859/152368036-082fb005-5a27-481d-8eb0-daa1797edaa7.png">


**1-Data unserstanding**
we started by data collection, using the selenium python library we scraped data from twitter we basically foccused on scrapping data in 3 different languages: Darija, French and English. Then we explored our dataset to understand it's specifities and caracteristics.

**2-Data preprocessing**
This is one of the most important steps in any modelisation probleme, data preprocessing plays a crucial role since the modelisation technics are not equipped to process non-structed data especially in our case, where we're dealing with textual data. well see more details further in the notebook.

**3-Modeling**
After getting our data ready, and compatible with machine learning algorithms inputs, we're ready tobuild our model, the challenge here is that we have several types of algorithms and we will have to chose which one preforms the best in our case. 

**4-Evaluation**
After building our models we move to evaluationg them using different technics.

After succesfully cleaning our dataset, we move to building the matrix how is it done?


#  **Vectorization**
To move on to the creation of machine learning models, we must first transform the text into a data matrix that corresponds to the processing by ML algorithms, while trying to minimizing the loss of information as much as possible.
each line in our dataset will represent the lines of our matrix hence we speak of a vector presentation, but in order to determine the features or the indexes we will use the countvectorizer.
The countvectorizer has many parameters to do indexation we have chosen to use the N-gram of letters.
Here's a schema of what we're going to do using n-gram

<img width="488" alt="all" src="https://user-images.githubusercontent.com/66137466/152540807-a13ec403-4444-47c0-a1fa-e9a47fae4717.PNG">
<img width="488" alt="all" src="https://user-images.githubusercontent.com/66137466/152540899-9d65fe47-08db-4633-9d1e-b3f9d2b163bf.PNG">


##  **1-Unigrams**

**vector presentation of languages using uni-gram**

<img width="518" alt="tale" src="https://user-images.githubusercontent.com/81523859/152369504-70624c94-83c2-45d3-90c6-5412c8f3cac7.png">

Let's take an exemple to understand what's going on:

Here's how does the countvectorizer work:


<img width="497" alt="tp1" src="https://user-images.githubusercontent.com/81523859/152370021-d44a3706-3dd9-493e-8038-7956077caf49.png">

Ps: in this exemple we're refering to the uni-gram parameter, it's pretty clear since the size of the constracted vector is equal to 32 which is the number of unique features in uni-gram!

##  **2-Bigrams**

**top bigrams (>1%) for each language**

<img width="518" alt="tale" src="https://user-images.githubusercontent.com/66137466/152541095-b5f32474-318f-4139-8572-3ba9fa890158.PNG">

Let's take an exemple to understand what's going on:

Here's how does the countvectorizer work:

<img width="497" alt="tp1" src="https://user-images.githubusercontent.com/66137466/152551408-6e32551f-aad4-46db-8310-491ad2f0b94a.PNG">
<img width="497" alt="tp1" src="https://user-images.githubusercontent.com/66137466/152541505-34024d5b-2ad5-4c27-a540-685982e38d4b.PNG">


Ps: in this exemple we're refering to the bi-gram parameter, the size of the constracted vector is equal to 941 which is the number of unique features in bi-grams!


##  **3-Top 1%  Mixture Uni-grams and Bi-grams**

**top Mixture (>1%) for each language**

<img width="518" alt="tale" src="https://user-images.githubusercontent.com/66137466/152536812-776498b7-0d00-4c43-9186-77562b227847.PNG">
<img width="518" alt="tale" src="https://user-images.githubusercontent.com/66137466/152536875-6f1562d6-effa-41a6-b86a-42132c658d08.PNG">
<img width="518" alt="tale" src="https://user-images.githubusercontent.com/66137466/152536944-ccc0bdb6-e430-4a83-8193-287e2bcd9fd9.PNG">
<img width="518" alt="tale" src="https://user-images.githubusercontent.com/66137466/152541602-9eef77e1-432f-4455-9633-763b7bd59f72.PNG">


Let's take an exemple to understand what's going on:

Here's how does the countvectorizer work:


<img width="497" alt="tp1" src="https://user-images.githubusercontent.com/66137466/152538093-28d961e5-f1f2-4a17-9bc3-6911e41d481a.PNG">
<img width="497" alt="tp1" src="https://user-images.githubusercontent.com/66137466/152538590-1fda1d28-fdfb-4b77-91f4-6a79b71be3c8.PNG">



Ps: in this exemple we're refering to the mixture parameter, the size of the constracted vector is equal to 471 .

##  **4-Top 60  Mixture Uni-grams and Bi-grams**

**top 60 Mixture  for each language**

<img width="518" alt="tale" src="https://user-images.githubusercontent.com/66137466/152538869-bd561bb0-9e97-4dc4-a003-455225cb5736.PNG">
<img width="518" alt="tale" src="https://user-images.githubusercontent.com/66137466/152551763-88d8b42b-b9e0-48b6-8c25-a298ab7445f9.PNG">

Let's take an exemple to understand what's going on:

Here's how does the countvectorizer work:

<img width="497" alt="tp1" src="https://user-images.githubusercontent.com/66137466/152539427-ab27a9c8-d714-42b3-81c8-5c491dd1a3b1.PNG">

#  **Models**

For this problem, we used 3 classification models:
 ### Naive Bayes Multinomial
 ### K Nearest Neighbor
 ### Logistic Regression
 
 **Result**:
 After applying k-fold cross validation, we found that Logistic regression using Top 1%  Mixture is the best model, because he was able to distinguish more or less between the languages.
 

#  **Deployement**


[You can reach the application on streamlit here.](https://share.streamlit.io/ikrambel22/language_detection/ikram-pull-branch/streamlitapi.py) 
