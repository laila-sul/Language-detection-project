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

#  **Building the matrix**
To move on to the creation of machine learning models, we must first transform the text into a data matrix that corresponds to the processing by ML algorithms, while trying to minimizing the loss of information as much as possible.
each line in our dataset will represent the lines of our matrix hence we speak of a vector presentation, but in order to determine the features or the indexes we will use the countvectorizer.
The countvectorizer has many parameters to do indexation we have chosen to use the N-gram of letters.
Here's a schema of what we're going to do using n-gram


<img width="341" alt="T2" src="https://user-images.githubusercontent.com/81523859/152369184-314d14c3-753f-40ec-8ae9-96e557aab168.png">

<img width="488" alt="all" src="https://user-images.githubusercontent.com/81523859/152369273-357b8037-6a15-4fc0-af60-de5eb4cec7ad.png">


#  **Exemple**


**vector presentation of languages using uni-gram**

<img width="518" alt="tale" src="https://user-images.githubusercontent.com/81523859/152369504-70624c94-83c2-45d3-90c6-5412c8f3cac7.png">

Let's take an exemple to understand what's going on:

Here's how does the countvectorizer work:


<img width="497" alt="tp1" src="https://user-images.githubusercontent.com/81523859/152370021-d44a3706-3dd9-493e-8038-7956077caf49.png">


Ps: in this exemple we're refering to the uni-gram parameter, it's pretty clear since the size of the constracted vector is equal to 32 which is the number of unique features in uni-gram!


