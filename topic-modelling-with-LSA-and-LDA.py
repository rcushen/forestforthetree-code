
# coding: utf-8

# *In this post, two typical NLP techniques are explored in relation to the problem of topic modelling. These are applied to the 'A Million News Headlines' dataset, which is a corpus of over one million news article headlines published by the ABC.*
# 
# # Introduction
# 
# ## Motivating the Problem
# 
# Topic modelling is a classic problem in the domain of natural language processing. Given a corpus of text $\mathcal{C}$ composed of $n$ documents, the goal is to uncover any latent topic structure that may exist within the data. In other words, we are seeking to discover the set of topics $\mathcal{T}$ with which a given series of documents are concerned, and having done so, sort these documents into the differing topic categories. Since we do not know a priori exactly what these topics are, this is an *unsupervised* problem: we have only unlabelled input data, and must determine the topic categories endogenously. In many respects, topic modelling is therefore quite similar to a clustering problem – by 'modelling topics', we are essentially just clustering the text headlines, where the clusters now have an additional interpretation as topic categories. The primary difference is that we are now working in some abstract 'word' space, rather than a more conventional Euclidean vector space in $\mathbb{R}^p$.
# 
# LSA and LDA represent two distinct approaches to solving this problem. Each draws upon very different mathematical procedures, and will have varying success depending upon the type of text data that is supplied as input. As we shall however see, both can be implemented in very similar ways, affording us ample scope for an exploration of their respective strengths and weaknesses.
# 
# ## The Dataset
# 
# ### Background
# 
# The dataset which we will be using for our exploration of topic modelling is the 'A Million News Headlines' corpus, published by the ABC. This is a set of over one million news article headlines taken from the Australian public broadcaster's news division, collected between 2003 and 2017. Further details can be found on [the Kaggle page](https://www.kaggle.com/therohk/million-headlines). In their raw form, the headlines are available simply as a series of text strings, accompanied by a publish date. Some examples are given below.
# 
# | Publish Date  | Headline      |
# | :-------------: |:-------------:|
# | 2003-02-19 | businesses should prepare for terrorist attacks |
# | 2004-05-12 | councillor to contest wollongong as independent    |
# | 2017-09-30 | san juan mayor mad as hell over us hurricane      |
# 
# More typically, topic modelling is conducted upon longer-form text data – perhaps entire news articles, or extracts from novels, or Wikipedia pages. For intuitive reasons, longer text samples are indeed preferable; more words means a richer snapshot of potential topics and a more diverse topic vocabulary. However, thanks to the succinct and lucid nature of news article headlines, we can still expect a robust kernel of semantic content, and paired with the massive number of datapoints available, it is unlikely that our analysis will suffer for this lack of depth.
# 
# ### Feature Construction
# 
# As with most forms of text analysis, the text data must first be preprocessed before it can be used as input to any algorithm (for the moment, we will disregard the publish date). In our case, this will mean converting each headline into a (very long) vector, where each entry corresponds to the number of occurences of a particular word. Effectively, this is equivalent to walking through the string with a dictionary in hand, adding a check next to each word in the dictionary as they appear. The resulting list of word counts then becomes the vector associated with that string.
# 
# $$ \text{''the cat in the hat''} \longrightarrow \begin{bmatrix} 1 \\ 0 \\ 2 \\ 0 \\ 0 \\ \vdots \\ 1 \\ 0 \end{bmatrix} \in \begin{bmatrix} \text{cat} \\ \text{fantastic} \\ \text{the} \\ \text{fox} \\ \text{machine} \\ \vdots \\ \text{in} \\ \text{chocolate} \end{bmatrix}$$
# 
# This approach is known as a **bag of words** representation, in that it reduces each text string to just a collection of word counts and disregards the actual ordering of words. Of course, encoding features in this way involves a loss of information, and a more sophisticated approach might attempt to recognise specific sequences of words – but for our initial implementation, this is an acceptable simplification.
# 
# *It is worth noting that this process of vectorisation has several variations. Rather than the naive approach taken above, a common alternative is [term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf–idf) or TFIDF. This computes the relative frequency of words in a document compared to the whole corpus, partially in attempt to counteract the greater significance that a longer document would have over a shorter one if raw counts were used. However, given that we are working with a corpus of text strings of relatively equal length—and moreover, strings which are all very short—the naive approach is perfectly reasonable.*
# 
# Vectorizing all $n$ headlines in this manner will ultimately yield an $n \times K$ **document-term matrix**, where each row corresponds to a headline and each column corresponds to a distinct word. The exact set of words chosen—i.e. our actual feature set—must be selected exogenously: the goal is to generate a diverse feature space while at the same time preventing the data from being too high-dimensional. For this reason, we typically use the top $M$ most frequent words in the corpus while also omitting a set of common 'stop words' – trivial prepositions and conjunctions (e.g. the, in, at, that, to). This 'middle band' of vocabulary should then give rise to a concise but meaningful set of features (words) that suffices to capture the key variation in the data. To recap: the table of one million headline text strings has been converted to a document-term matrix $D$, which can now be used as input to our topic model.
# 
# $$ \begin{bmatrix} \text{the cat in the hat} 
# \\ \text{green eggs and ham}
# \\ \vdots
# \\ \text{horton hears a who}
# \end{bmatrix}
# \longrightarrow
# \underbrace{
# \begin{bmatrix} 
# 1 & 0 & 2 & 0 & 0 & \cdots & 1 & 0 \\
# 0 & 1 & 0 & 0 & 1 & \cdots & 0 & 1 \\
# \vdots & \vdots & \vdots & \vdots & \vdots & & \vdots & \vdots \\
# 0 & 0 & 0 & 1 & 0 & \cdots & 0 & 0
# \end{bmatrix}
# }_{D} $$
# 
# ## A Brief Review of LSA and LDA
# 
# Before embarking upon our analysis, let's briefly outline the two techniques that are to be employed. The first of these is Latent Semantic Analysis, more typically referred to as LSA. At its heart, this is fundamentally just a matrix factorisation procedure: given a document-term matrix $D$, the algorithm first performs singular value decomposition, and then truncates all but the $r$ largest singular values. Instead of the usual SVD factorisation,
#     $$ D = U \Sigma V^T $$
# we therefore use
#     $$ T = U \hat{\Sigma} V^T $$
# where $\hat{\Sigma}$ is the matrix of singular values $\Sigma$ with all but the $r$ largest singular values set to zero. The resulting matrix $T$ is thus of reduced rank, and is referred to as the **topic matrix**. Each of its $n$ rows will represent a document, and each of the first $r$ columns will correspond to a topic; the $(i, j)$th entry can then be considered a measure of the presence of topic $j$ in document $i$. To actually sort a document into a topic category, we simply take the $\arg \max$ of each row, as this will correspond to the most strongly-represented topic. Note too that $r$ is a parameter here, and must be supplied as an input to the algorithm. In other words, the number of topic categories must be provided exogenously – not all of the decision are made internally by the algorithm.
# 
# Meanwhile, Latent Dirichilet Allocation takes an entirely different approach. Rather than a matrix decomposition, it is instead a generative probabilistic process. It views documents—in this case, headlines—as probability distributions over latent topics, and these topics to be probability distributions over words. As such, the technique supposes that headlines are generated according to the following:
# 1. Pick a number of words in the headline.
# 2. Choose a topic mixture for the headline, over a fixed number of topics.
# 3. Generate the words in the headline by picking a topic from the headline’s multinomial distribution, and then picking a word based upon the topic’s multinomial distribution.
# 
# To actually sort the headlines into topic clusters, LDA then works backwards: starting from Dirichilet priors, variational Bayes methods are used to infer the latent distributional parameters, which then characterise the differing topics. As with LSA, the actual number of topics $r$ must be supplied as a hyperparameter here. The output will then once again be in the form of a topic matrix $T$, though each of the rows are now effectively a probability distribution defined over the topics for each document. As such, the $(i,j)$th entry of this topic matrix can now be interpreted as a probability that headline $i$ belongs to topic $j$ (or rather, as a proportion of the words in the headline which fall into topic $j$). Hence by taking the $\arg \max$ of each row, we again obtain an estimated topic category for each headline.

# # Analysis
# 
# Having set the scene, we can now dive into developing our topic models. All relevant code is featured below as extracts from a Jupyter Notebook, though a more workable Python script can be found the GitHub page linked above.
# 
# ## Exploratory Data Analysis
# 
# As usual, it is prudent to begin with some basic EDA. After the usual import statements...

# In[1]:


import numpy as np
import pandas as pd
from IPython.display import display
from collections import Counter
import ast

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.patheffects as path_effects

get_ipython().magic(u'matplotlib inline')

datafile = '../../data/abcnews-date-text.csv'
raw_data = pd.read_csv(datafile, parse_dates=[0], infer_datetime_format=True)

reindexed_data = raw_data['headline_text']
reindexed_data.index = raw_data['publish_date']

display(raw_data.head())


# ...we first develop a list of the top words used across all one million headlines, giving us a glimpse into the core vocabulary of the source data. Stop words are omitted here to avoid any trivial prepositions and conjunctions.

# In[2]:


# Define helper functions
def get_top_n_words(n_top_words, count_vectorizer, text_data):
    '''returns a tuple of a list of the top n words in a sample and a list of their accompanying counts, given a CountVectorizer object and text sample'''
    vectorized_headlines = count_vectorizer.fit_transform(text_data.as_matrix())
    
    vectorized_total = np.sum(vectorized_headlines, axis=0)
    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)
    word_values = np.flip(np.sort(vectorized_total)[0,:],1)
    
    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))
    for i in range(n_top_words):
        word_vectors[i,word_indices[0,i]] = 1

    words = [word[0].encode('ascii').decode('utf-8') for word in count_vectorizer.inverse_transform(word_vectors)]

    return (words, word_values[0,:n_top_words].tolist()[0])


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words='english')
words, word_values = get_top_n_words(n_top_words=20, count_vectorizer=count_vectorizer, text_data=reindexed_data)

fig, ax = plt.subplots(figsize=(12,5))
ax.bar(range(len(words)), word_values)
ax.set_xticks(range(len(words)));
ax.set_xticklabels(words, rotation='vertical');
ax.set_title('Top Words in Headlines Corpus');


# Even from this initial distribution, several topics are hinted at. Headlines describing crime and violence seem to appear frequently ('police', 'court', 'crash', 'death' and 'charged'), and politics also demonstrates a presence ('govt', 'council', 'nsw', 'australia', 'qld', 'wa'). Of course, these assocations are a little tenuous, but manifest structure of this kind is certainly encouraging for our later application of LSA and LDA.
# 
# Next we generate a histogram of headline word lengths, and use [part-of-speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging) to understand the types of words used across the corpus. This can be done using the TextBlob library, which incorporates a speech tagging function ```pos_tags``` that can be used to generate a list of tagged words for each headline. A complete list of such word tags is available [here](https://www.clips.uantwerpen.be/pages/MBSP-tags).

# In[4]:


from textblob import TextBlob

while True:
    try:
        tagged_headlines = pd.read_csv('../../data/abcnews-pos-tagged.csv', index_col=0)
        word_counts = [] 
        pos_counts = {}

        for headline in tagged_headlines[u'tags']:
            headline = ast.literal_eval(headline)
            word_counts.append(len(headline))
            for tag in headline:
                if tag[1] in pos_counts:
                    pos_counts[tag[1]] += 1
                else:
                    pos_counts[tag[1]] = 1

    except IOError:
        tagged_headlines = [TextBlob(reindexed_data[i]).pos_tags for i in range(reindexed_data.shape[0])]

        tagged_headlines = pd.DataFrame({'tags':tagged_headlines})
        tagged_headlines.to_csv('../../data/abcnews-pos-tagged.csv')
        continue
    break

print 'Total number of words: ', np.sum(word_counts)
print 'Mean number of words per headline: ', np.mean(word_counts)


# In[5]:


fig, ax = plt.subplots(figsize=(12,5))
ax.hist(word_counts, bins=range(1,14), normed=1)
ax.set_title('Headline Word Lengths')
ax.set_xticks(range(1,14))
ax.set_xlabel('Number of Words')
y = mlab.normpdf( np.linspace(0,14,50), np.mean(word_counts), np.std(word_counts))
l = ax.plot(np.linspace(0,14,50), y, 'r--', linewidth=1)


# In[6]:


pos_sorted_types = sorted(pos_counts, key=pos_counts.__getitem__, reverse=True)
pos_sorted_counts = sorted(pos_counts.values(), reverse=True)

fig, ax = plt.subplots(figsize=(12,5))
ax.bar(range(len(pos_counts)), pos_sorted_counts)
ax.set_xticks(range(len(pos_counts)));
ax.set_xticklabels(pos_sorted_types, rotation='vertical');
ax.set_title('Part-of-Speech Tagging for Headlines Corpus');
ax.set_xlabel('Type of Word');


# From the histogram, we can see that the headlines are approximately normally distributed in length, with most composed of between five and eight words. This certainly validates our naive use of word counts over TFIDF! Meanwhile, the results of the POS tagging are also encouraging: a significant portion of the vocabulary in the corpus are nouns (NN, NNS), which are much more likely to be topic-specific than any other part of speech.

# ## Topic Modelling
# 
# We now turn to apply the two topic modelling algorithms. To facilitate computation, we start by using just a small subsample of the data.

# ### Preprocessing
# The only preprocessing step required in our case is feature construction, discussed above. This can be done using the ```CountVectorizer``` object from SKLearn, which takes as input a list of text strings and yields an $n×K$ document-term matrix. In this case, we set $K = 40,000$, denoting the $40,000$ most common words across the $n$ headlines in our sample (less stop words).

# In[7]:


count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
text_sample = reindexed_data.sample(n=10000, random_state=0).as_matrix()

print 'Headline before vectorization: ', text_sample[0]

document_term_matrix = count_vectorizer.fit_transform(text_sample)

print 'Headline after vectorization: \n', document_term_matrix[0]


# It is worth noting here that the document-term matrix has been stored by SciPy as a sparse matrix. This is a key idiosyncracy that often arises in NLP analysis – given we are working with such high-dimensional features, it is crucial that data be stored as efficiently as possible (maintaining $1,000,000 \times 40,000 = 40 \ \text{billion}$ floating point numbers in memory is certainly not feasible!). Luckily, the bag of words model means that most of the entries in a document-term matrix are zero, since most pieces of text are composed of only a small fraction of the total allowed vocabulary. As such the document-term matrix $D$ is highly sparse, and this can be taken advantage of when performing matrix operations.
# 
# Thus we have constructed our input data, ```document_term_matrix```, and can now actually implement a topic modelling algorithm. Both LSA and LDA will take our document-term matrix as input and yield an $n \times N$ topic matrix as output, where $N$ is the number of topic categories (which we supply as a parameter, as discussed above). For the moment, 8 would seem to be a reasonable number; any more and we would risk being overly granular with the topic categories, any fewer and we would be grouping together potentially unrelated content. Nonetheless, this can be experimented with later.

# In[8]:


n_topics = 8


# ### Latent Semantic Analysis
# Let's start by experimenting with LSA. This can be easily implemented using the ```TruncatedSVD``` class in SKLearn.

# In[9]:


from sklearn.decomposition import TruncatedSVD

lsa_model = TruncatedSVD(n_components=n_topics) # Instantiates the LSA model
lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix) # Runs the truncated SVD


# Taking the $\arg \max$ of each headline in this topic matrix will give the predicted topics of each headline in the sample. We can then sort these into counts of each topic.

# In[10]:


# Define helper functions
def get_keys(topic_matrix):
    '''returns an integer list of predicted topic categories for a given topic matrix'''
    keys = []
    for i in range(topic_matrix.shape[0]):
        keys.append(topic_matrix[i].argmax())
    return keys

def keys_to_counts(keys):
    '''returns a tuple of topic categories and their accompanying magnitudes for a given list of keys'''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)


# In[11]:


lsa_keys = get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)


# However, these topic categories are in and of themselves a little meaningless. In order to better characterise them, it will be helpful to find the most frequent words in each.

# In[12]:


# Define helper functions
def get_top_n_words(n, keys, document_term_matrix, count_vectorizer):
    '''returns a list of n_topic strings, where each string contains the n most common words in a predicted category, in order'''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)   
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))         
    return top_words


# In[13]:


top_n_words_lsa = get_top_n_words(10, lsa_keys, document_term_matrix, count_vectorizer)

for i in range(len(top_n_words_lsa)):
    print "Topic {}: ".format(i), top_n_words_lsa[i]


# Hence we have converted our initial small sample of headlines into a list of predicted topic categories, where each category is characterised by its most frequent words. The relative magnitudes of each of these categories can then be easily visualised though use of a bar chart.

# In[14]:


top_3_words = get_top_n_words(3, lsa_keys, document_term_matrix, count_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]

fig, ax = plt.subplots(figsize=(12,5))
ax.bar(lsa_categories, lsa_counts)
ax.set_xticks(lsa_categories);
ax.set_xticklabels(labels, rotation='vertical');
ax.set_title('LSA Topic Category Counts');


# Several insights are apparent here. It is clear that LSA has successfully delineated some intuitive topic categories: Topics 0 and 2 appear to be 'Crime', Topic 4 could be 'Legal Proceedings', Topic 5 'National Politics', and Topic 2 'World Affairs'. The distribution of topics is also non-uniform, suggesting—as would be expected—that certain topics are more prevalent than others in news reporting. Both of these outcomes therefore represent confirmations of a prior belief, which is encouraging.
# 
# However, this output does not provide a great point of comparison with other clustering algorithms. In order to properly contrast LSA with LDA we instead use a dimensionality-reduction technique called $t$-SNE, which will also serve to better illuminate the success of the clustering process. $t$-SNE takes each of the $n$ vectors from the topic matrix and projects them from $N$ dimensions to two dimensions, such that they can be visualised in a scatter plot. The way $t$-SNE functions is a little opaque, and it is beyond the scope of this post to discuss its workings in detail (see [this article](https://distill.pub/2016/misread-tsne/) for a great exploration of its properties), though it has been proven to yield great results on non-linear datasets such as our topic matrix. It can also be implemented in SKLearn, though it has a relatively long runtime even on our small sample.

# In[19]:


from sklearn.manifold import TSNE

tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_matrix)


# Now that we have reduced these ```n_topics```-dimensional vectors to two-dimensional representations, we can plot the clusters. Before doing so however, it will be useful to derive the centroid location of each topic, so as to better contextualise our visualisation.

# In[20]:


# Define helper functions
def get_mean_topic_vectors(keys, two_dim_vectors):
    '''returns a list of centroid vectors from each predicted topic category'''
    mean_topic_vectors = []
    for t in range(n_topics):
        articles_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                articles_in_that_topic.append(two_dim_vectors[i])    
        
        articles_in_that_topic = np.vstack(articles_in_that_topic)
        mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
        mean_topic_vectors.append(mean_article_in_that_topic)
    return mean_topic_vectors


# In[27]:


colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])
colormap = colormap[:n_topics]


# All that remains is to plot the clustered headlines. Also included are the top three words in each cluster, which are placed at the centroid for that topic.

# In[59]:


fig, ax = plt.subplots(figsize=(12,12))
ax.scatter(x=tsne_lsa_vectors[:,0], y=tsne_lsa_vectors[:,1], color=colormap[lsa_keys])
ax.set_title("t-SNE Clustering of {} LSA Topics".format(n_topics));

for t in range(n_topics):
    ax.text(x=lsa_mean_topic_vectors[t][0], y=lsa_mean_topic_vectors[t][1], 
                  s=top_3_words_lsa[t], color=colormap[t], path_effects=[path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])


# Evidently, this is a bit a of a failed result. While the topic categories derived above appeared to be generally coherent, the scatterplot makes it clear that separation between these categories is poor. Such a lack of clarity was in fact hinted at earlier, when we generated the top ten words in each topic: several words were frequent in multiple topics, suggesting a failure to distinguish between content. And perhaps most obviously, the centroid locations for each topic cluster are completely nonsensical. Nonetheless, it is difficult to tell whether this can be attributed to the LSA decomposition or instead the $t$-SNE dimensionality reduction process. Let's move forward and experiment with another topic modelling technique to see if we can improve upon this outcome.

# ### Latent Dirichilet Allocation
# We now repeat this process using LDA instead of LSA. Once again, SKLearn provides a very simple implementation via its ```LatentDirichletAllocation``` class.

# In[15]:


from sklearn.decomposition import LatentDirichletAllocation

lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=0, verbose=0) # Instatiate the LDA class
lda_topic_matrix = lda_model.fit_transform(document_term_matrix) # Runs the LDA process


# Once again, we take the $\arg \max$ of each entry in the topic matrix to obtain the predicted topic category for each headline. These topic categories can then be characterised by their most frequent words.
# 

# In[16]:


lda_keys = get_keys(lda_topic_matrix)
lda_categories, lda_counts = keys_to_counts(lda_keys)


# In[33]:


top_n_words_lda = get_top_n_words(10, lda_keys, document_term_matrix, count_vectorizer)

for i in range(len(top_n_words_lda)):
    print "Topic {}: ".format(i), top_n_words_lda[i]


# The relative topic compositions of the sample are then illustated with a barchart.

# In[34]:


top_3_words = get_top_n_words(3, lda_keys, document_term_matrix, count_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lda_categories]

fig, ax = plt.subplots(figsize=(12,5))
ax.bar(lda_categories, lda_counts)
ax.set_xticks(lda_categories);
ax.set_xticklabels(labels, rotation='vertical');
ax.set_title('LDA Topic Category Counts');


# Already, we see starkly different results from those generated by LSA. The topics that have been delineated are distinct, and are in fact less coherent – no obvious interpretations jump out, other than perhaps 'Crime' for Topic 3 and Topic 4. Moreover, the distribution of topics is much flatter. This is likely a consequence of the variational Bayes algorithm, which begins with equal priors across all $N$ categories and only gradually updates them as it iterates through the corpous. While such a flat distribution may not seem especially significant, it may reflect deeper problems with the algorithm; in effect, LDA is telling us that the headlines are an almost-equal mix of the $N$ categories, which we would not necessarily expect to be an intuitive result.
# 
# Nonetheless, in order to properly compare LDA with LSA, we again take this topic matrix and project it into two dimensions with $t$-SNE.

# In[35]:


from sklearn.manifold import TSNE

tsne_lda_model = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, verbose=1, random_state=0, angle=0.75)
tsne_lda_vectors = tsne_lda_model.fit_transform(lda_topic_matrix)


# In[60]:


fig, ax = plt.subplots(figsize=(12,12))
ax.scatter(x=tsne_lda_vectors[:,0], y=tsne_lda_vectors[:,1], color=colormap[lda_keys])
ax.set_title("t-SNE Clustering of {} LDA Topics".format(n_topics));

for t in range(n_topics):
    ax.text(x=lda_mean_topic_vectors[t][0], y=lda_mean_topic_vectors[t][1], 
                  s=top_3_words_lda[t], color=colormap[t], path_effects=[path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])


# This now appears a much better result! Controlling for $t$-SNE, it would seem that LDA has had a lot more succcess than LSA in separating out the topic categories. There is much less overlap between the clusters, and each topic has been sorted into an almost-contiguous region. Clearly, LDA is capable of generating a much more intelligible projection of the headlines into topic space (most likely because this topic space is indeed a probability space), but this must certainly be taken with a grain of salt given the lack of coherence of these topic clusters discussed above.

# # Conclusions
# 
# Each of the two algorithms has had varying degrees of success. While LSA generated a more coherent topic-set and a more reasonable topic distribution, it did not succeed in attaining great separation between these clusters. LDA instead achieved nearly the opposite: separation between topics was very good, but the topics derived were not especially intelligible, and their distribution seemed unlikely. As usual, we find that the No Free Lunch principle holds. Topic modelling is a fundamentally hard problem!
# 
# Nonetheless, our results are broadly encouraging. The potential for delineation and coherence of topics in the headlines data has been made apparent, and several fundamental properties of the dataset have been illuminated. There may of course be scope for improvement: hyperparameters could be tuned (the number of topics $N$ could be varied to try and obtain a more interpretable topic set, LDA's Dirichilet priors could be tailored to better suit the data, $t$-SNE could be played around with to try and improve projections from topic space to two-dimesions, and so on), and more time could also be spent on feature construction, to perhaps further reduce down the size of the vocabulary set. But regardless, the above explorations establish a strong foundation for any further analyses of the headlines dataset – and moreover, clearly demonstrate the viability of topic modelling on an ostensibly novel data format.
