import numpy as np
import lda
from sklearn.feature_extraction.text import CountVectorizer

## single category
lines = open('./single_category/titles.csv', encoding="utf8").readlines()
data = CountVectorizer(input=lines, stop_words=None)
X = data.fit_transform(lines)
X = X.toarray()
keep = np.where(np.sum(X, axis = 1) !=0)[0]
X = X[keep]
model = lda.LDA(n_topics=50, n_iter=1500, random_state=1)
model.fit(X)
topic_word = model.topic_word_
doc_topic = model.doc_topic_
np.save("./single_category/doc_topic_titles.npy", doc_topic)
stars = np.load("./single_category/stars.npy")[keep]
np.save("./single_category/doc_topic_titles_y.npy",stars)

place = ['begin','middle','end']
for loc in place:
    lines = open('./single_category/reviews_'+loc+'.csv', encoding="utf8").readlines()
    data = CountVectorizer(input=lines, stop_words=None)
    X = data.fit_transform(lines)
    X = X.toarray()
    X = X[keep]
    model = lda.LDA(n_topics=50, n_iter=1500, random_state=1)
    model.fit(X)
    doc_topic = model.doc_topic_
    np.save("./single_category/doc_topic_reviews_"+loc+".npy", doc_topic)
    stars = np.load("./single_category/stars.npy")[keep]
    np.save("./single_category/doc_topic_reviews_"+loc+"_y.npy",stars)



## individual category titles
type_of_products = ['TVs', 'cameras','laptops','mobilephone','tablets','video_surveillance']
for type in type_of_products:
    read_filename = "./ind_category/titles_"+type+'.csv'
    lines = open(read_filename, encoding="utf8").readlines()
    data = CountVectorizer(input=lines, stop_words=None)
    X = data.fit_transform(lines)
    X = X.toarray()
    keep = np.where(np.sum(X, axis = 1) !=0)[0]
    X1 = X[keep]
    model = lda.LDA(n_topics=50, n_iter=1500, random_state=1)
    model.fit(X1)
    #topic_word = model.topic_word_
    doc_topic = model.doc_topic_
    save_filename = "./ind_category/doc_topic_titles_"+type+".npy"
    np.save(save_filename, doc_topic)
    read_starname = "./ind_category/stars_"+type+'.npy'
    save_starname ="./ind_category/doc_topic_titles_y_"+type+".npy"
    stars = np.load(read_starname)[keep]
    np.save(save_starname,stars)

    read_filename = "./ind_category/reviews_begin_" + type + '.csv'
    lines = open(read_filename, encoding="utf8").readlines()
    data = CountVectorizer(input=lines, stop_words=None)
    X = data.fit_transform(lines)
    X = X.toarray()
    X2 = X[keep]
    model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
    model.fit(X2)
    # topic_word = model.topic_word_
    doc_topic = model.doc_topic_
    save_filename = "./ind_category/doc_topic_reviews_begin_" + type + ".npy"
    np.save(save_filename, doc_topic)
    read_starname = "./ind_category/stars_" + type + '.npy'
    save_starname = "./ind_category/doc_topic_reviews_begin_y_" + type + ".npy"
    stars = np.load(read_starname)[keep]
    np.save(save_starname, stars)

    read_filename = "./ind_category/reviews_middle_" + type + '.csv'
    lines = open(read_filename, encoding="utf8").readlines()
    data = CountVectorizer(input=lines, stop_words=None)
    X = data.fit_transform(lines)
    X = X.toarray()
    X3 = X[keep]
    model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
    model.fit(X3)
    # topic_word = model.topic_word_
    doc_topic = model.doc_topic_
    save_filename = "./ind_category/doc_topic_reviews_middle_" + type + ".npy"
    np.save(save_filename, doc_topic)
    read_starname = "./ind_category/stars_" + type + '.npy'
    save_starname = "./ind_category/doc_topic_reviews_middle_y_" + type + ".npy"
    stars = np.load(read_starname)[keep]
    np.save(save_starname, stars)

    read_filename = "./ind_category/reviews_end_" + type + '.csv'
    lines = open(read_filename, encoding="utf8").readlines()
    data = CountVectorizer(input=lines, stop_words=None)
    X = data.fit_transform(lines)
    X = X.toarray()
    X4 = X[keep]
    model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
    model.fit(X4)
    # topic_word = model.topic_word_
    doc_topic = model.doc_topic_
    save_filename = "./ind_category/doc_topic_reviews_end_" + type + ".npy"
    np.save(save_filename, doc_topic)
    read_starname = "./ind_category/stars_" + type + '.npy'
    save_starname = "./ind_category/doc_topic_reviews_end_y_" + type + ".npy"
    stars = np.load(read_starname)[keep]
    np.save(save_starname, stars)




# ## individual category reviews
# type_of_products = ['TVs', 'cameras','laptops','mobilephone','tablets','video_surveillance']
# for type in type_of_products:
#     read_filename = "./ind_category/reviews_"+type+'.csv'
#     lines = open(read_filename, encoding="utf8").readlines()
#     data = CountVectorizer(input=lines, stop_words=None)
#     X = data.fit_transform(lines)
#     X = X.toarray()
#     keep = np.where(np.sum(X, axis = 1) !=0)[0]
#     X = X[keep]
#     model = lda.LDA(n_topics=100, n_iter=1500, random_state=1)
#     model.fit(X)
#     #topic_word = model.topic_word_
#     doc_topic = model.doc_topic_
#     save_filename = "./ind_category/doc_topic_reviews_"+type+".npy"
#     np.save(save_filename, doc_topic)
#     read_starname = "./ind_category/stars_"+type+'.npy'
#     save_starname ="./ind_category/doc_topic_reviews_y_"+type+".npy"
#     stars = np.load(read_starname)[keep]
#     np.save(save_starname,stars)

lines = open('./single_category/titles.csv', encoding="utf8").readlines()
data = CountVectorizer(input=lines, stop_words=None)
X = data.fit_transform(lines)
X = X.toarray()
keep1 = np.where(np.sum(X, axis = 1) !=0)[0]

lines = open('./single_category/reviews_begin.csv', encoding="utf8").readlines()
data = CountVectorizer(input=lines, stop_words=None)
X = data.fit_transform(lines)
X = X.toarray()
keep2 = np.where(np.sum(X, axis = 1) !=0)[0]

lines = open('./single_category/reviews_middle.csv', encoding="utf8").readlines()
data = CountVectorizer(input=lines, stop_words=None)
X = data.fit_transform(lines)
X = X.toarray()
keep3 = np.where(np.sum(X, axis = 1) !=0)[0]

lines = open('./single_category/reviews_end.csv', encoding="utf8").readlines()
data = CountVectorizer(input=lines, stop_words=None)
X = data.fit_transform(lines)
X = X.toarray()
keep4 = np.where(np.sum(X, axis = 1) !=0)[0]

keep = np.intersect1d(np.intersect1d(np.intersect1d(keep1, keep2), keep3), keep4)
