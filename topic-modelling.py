import numpy as np
import pandas as pd
from IPython.display import display
from collections import Counter
import ast

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.patheffects as path_effects

get_ipython().run_line_magic('matplotlib', 'inline')

datafile = 'data/abcnews-date-text.csv'
raw_data = pd.read_csv(datafile, parse_dates=[0], infer_datetime_format=True)

reindexed_data = raw_data['headline_text']
reindexed_data.index = raw_data['publish_date']

display(raw_data.head())

# Define helper functions
def get_top_n_words(n_top_words, count_vectorizer, text_data):
    '''returns a tuple of a list of the top n words in a sample and 
    a list of their accompanying counts, given a CountVectorizer object and text sample'''
    vectorized_headlines = count_vectorizer.fit_transform(text_data.as_matrix())
    
    vectorized_total = np.sum(vectorized_headlines, axis=0)
    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)
    word_values = np.flip(np.sort(vectorized_total)[0,:],1)
    
    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))
    for i in range(n_top_words):
        word_vectors[i,word_indices[0,i]] = 1

    words = [word[0].encode('ascii').decode('utf-8') for word in 
             count_vectorizer.inverse_transform(word_vectors)]

    return (words, word_values[0,:n_top_words].tolist()[0])



from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(stop_words='english')
words, word_values = get_top_n_words(n_top_words=20, 
                                     count_vectorizer=count_vectorizer, 
                                     text_data=reindexed_data)

fig, ax = plt.subplots(figsize=(12,5))
ax.bar(range(len(words)), word_values)
ax.set_xticks(range(len(words)));
ax.set_xticklabels(words, rotation='vertical');
ax.set_title('Top Words in Headlines Corpus');

plt.savefig('output/topwords.png', dpi=600, bbox_inches='tight')


from textblob import TextBlob

while True:
    try:
        tagged_headlines = pd.read_csv('data/abcnews-pos-tagged.csv', 
                                       index_col=0)
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
        tagged_headlines = [TextBlob(reindexed_data[i]).pos_tags 
                            for i in range(reindexed_data.shape[0])]

        tagged_headlines = pd.DataFrame({'tags':tagged_headlines})
        tagged_headlines.to_csv('data/abcnews-pos-tagged.csv')
        continue
    break

print('Total number of words: ', np.sum(word_counts))
print('Mean number of words per headline: ', np.mean(word_counts))

fig, ax = plt.subplots(figsize=(12,5))
ax.hist(word_counts, bins=range(1,14), normed=1)
ax.set_title('Headline Word Lengths')
ax.set_xticks(range(1,14))
ax.set_xlabel('Number of Words')
y = mlab.normpdf( np.linspace(0,14,50), np.mean(word_counts), np.std(word_counts))
l = ax.plot(np.linspace(0,14,50), y, 'r--', linewidth=1)
plt.savefig('output/wordhist.png', dpi=600, bbox_inches='tight')


pos_sorted_types = sorted(pos_counts, key=pos_counts.__getitem__, reverse=True)
pos_sorted_counts = sorted(pos_counts.values(), reverse=True)

fig, ax = plt.subplots(figsize=(12,5))
ax.bar(range(len(pos_counts)), pos_sorted_counts)
ax.set_xticks(range(len(pos_counts)));
ax.set_xticklabels(pos_sorted_types, rotation='vertical');
ax.set_title('Part-of-Speech Tagging for Headlines Corpus');
ax.set_xlabel('Type of Word');
plt.savefig('output/postags.png', dpi=600, bbox_inches='tight')


count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
text_sample = reindexed_data.sample(n=10000, random_state=0).as_matrix()

print('Headline before vectorization: {}'.format(text_sample[0]))

document_term_matrix = count_vectorizer.fit_transform(text_sample)

print('Headline after vectorization: \n {}'.format(document_term_matrix[0]))

n_topics = 8


from sklearn.decomposition import TruncatedSVD

lsa_model = TruncatedSVD(n_components=n_topics) # Instantiates the LSA model
lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix) # Runs the truncated SVD

# Define helper functions
def get_keys(topic_matrix):
    '''returns an integer list of predicted topic categories 
    for a given topic matrix'''
    keys = []
    for i in range(topic_matrix.shape[0]):
        keys.append(topic_matrix[i].argmax())
    return keys

def keys_to_counts(keys):
    '''returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys'''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)


lsa_keys = get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)


# Define helper functions
def get_top_n_words(n, keys, document_term_matrix, count_vectorizer):
    '''returns a list of n_topic strings, where each string contains 
    the n most common words in a predicted category, in order'''
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


# In[33]:


top_n_words_lsa = get_top_n_words(10, lsa_keys, 
                                  document_term_matrix, count_vectorizer)

for i in range(len(top_n_words_lsa)):
    print('Topic {}: {}'.format(i, top_n_words_lsa[i]))


top_3_words = get_top_n_words(3, lsa_keys, document_term_matrix, count_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]

fig, ax = plt.subplots(figsize=(12,5))
ax.bar(lsa_categories, lsa_counts)
ax.set_xticks(lsa_categories);
ax.set_xticklabels(labels, rotation='vertical');
ax.set_title('LSA Topic Category Counts');
plt.savefig('output/lsadist.png', dpi=600, bbox_inches='tight')


from sklearn.manifold import TSNE

tsne_lsa_model = TSNE(n_components=2, perplexity=50, 
                      learning_rate=100, n_iter=2000, verbose=1, 
                      random_state=0, angle=0.75)
tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_matrix)

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


# In[37]:


colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5" ])
colormap = colormap[:n_topics]


fig, ax = plt.subplots(figsize=(12,12))
ax.scatter(x=tsne_lsa_vectors[:,0], y=tsne_lsa_vectors[:,1], 
           color=colormap[lsa_keys])
ax.set_title("t-SNE Clustering of {} LSA Topics".format(n_topics));

lsa_mean_topic_vectors = get_mean_topic_vectors(lsa_keys, tsne_lsa_vectors)
top_3_words_lsa = get_top_n_words(3, lsa_keys, 
                                  document_term_matrix, count_vectorizer)

for t in range(n_topics):
    ax.text(x=lsa_mean_topic_vectors[t][0], y=lsa_mean_topic_vectors[t][1], 
            s=top_3_words_lsa[t], color=colormap[t], 
            path_effects=[path_effects.Stroke(linewidth=3, foreground='black'), 
                          path_effects.Normal()])

plt.savefig('output/lsaclusters.png', dpi=600, bbox_inches='tight')


from sklearn.decomposition import LatentDirichletAllocation

lda_model = LatentDirichletAllocation(n_components=n_topics, 
                                      learning_method='online', random_state=0, verbose=0)
lda_topic_matrix = lda_model.fit_transform(document_term_matrix)


lda_keys = get_keys(lda_topic_matrix)
lda_categories, lda_counts = keys_to_counts(lda_keys)


top_n_words_lda = get_top_n_words(10, lda_keys, document_term_matrix, count_vectorizer)

for i in range(len(top_n_words_lda)):
    print('Topic {}: {}'.format(i, top_n_words_lda[i]))


top_3_words = get_top_n_words(3, lda_keys, 
                              document_term_matrix, count_vectorizer)
labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lda_categories]

fig, ax = plt.subplots(figsize=(12,5))
ax.bar(lda_categories, lda_counts)
ax.set_xticks(lda_categories);
ax.set_xticklabels(labels, rotation='vertical');
ax.set_title('LDA Topic Category Counts');
plt.savefig('output/ldadist.png', dpi=600, bbox_inches='tight')

from sklearn.manifold import TSNE

tsne_lda_model = TSNE(n_components=2, perplexity=50, 
                      learning_rate=100, n_iter=2000, verbose=1, 
                      random_state=0, angle=0.75)
tsne_lda_vectors = tsne_lda_model.fit_transform(lda_topic_matrix)


fig, ax = plt.subplots(figsize=(12,12))
ax.scatter(x=tsne_lda_vectors[:,0], y=tsne_lda_vectors[:,1], 
           color=colormap[lda_keys])
ax.set_title("t-SNE Clustering of {} LDA Topics".format(n_topics));

lda_mean_topic_vectors = get_mean_topic_vectors(lda_keys, tsne_lda_vectors)
top_3_words_lda = get_top_n_words(3, lda_keys, 
                                  document_term_matrix, count_vectorizer)

for t in range(n_topics):
    ax.text(x=lda_mean_topic_vectors[t][0], y=lda_mean_topic_vectors[t][1], 
            s=top_3_words_lda[t], color=colormap[t], 
            path_effects=[path_effects.Stroke(linewidth=3, foreground='black'), 
                          path_effects.Normal()])
    
plt.savefig('output/ldaclusters.png', dpi=600, bbox_inches='tight')
