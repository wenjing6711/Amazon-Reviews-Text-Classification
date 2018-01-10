import gensim
import numpy as np
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
vocabulary = list(model.wv.vocab.keys())
word_dict = {}
for i in range(len(vocabulary)):
    word_dict[vocabulary[i]] = i
def get_word_index_and_label(x_text,y, doc_len, vocabulary, word_dict):
    x_id = np.zeros((len(x_text), doc_len))
    for row in range(len(x_text)):
        print(row)
        doc = x_text[row].split()
        index =[]
        for w in doc:
            if w in vocabulary:
                index.append(word_dict[w])
        index = np.array(index)
        if len(index) > doc_len:
            x_id[row,:] = index[0:doc_len]
        else:
            x_id[row,0:len(index)]= index
    new_y = np.zeros((len(y),2))
    for i in range(len(y)):
        if y[i] > 3:
            new_y[i,1] =1
        else:
            new_y[i,0]=1
    new_y2 = np.zeros((len(y),5))
    for i in range(len(y)):
        new_y2[i, y[i]-1] = 1
    return x_id, new_y,  new_y2

# ## one_category
# f = open("./single_category/reviews.csv")
# lines = f.readlines()
# y = np.load("./single_category/stars.npy")
# X, y_2_class, y_5_class = get_word_index_and_label(lines, y, 150, vocabulary, word_dict)
# train_index = np.random.choice(len(lines),  int(len(lines)*0.8),replace= False)
# test_index = np.setdiff1d(range(len(lines)), train_index)
# X_train = X[train_index]
# y_2_train = y_2_class[train_index]
# X_test = X[test_index]
# y_2_test = y_2_class[test_index]
# np.save("X_2_train.npy", X_train)
# np.save("y_2_train.npy", y_2_train)
# np.save("X_2_test.npy", X_test)
# np.save("y_2_test.npy", y_2_test)
# y_5_train = y_5_class[train_index]
# y_5_test = y_5_class[test_index]
# np.save("y_5_train.npy", y_5_train)
# np.save("y_5_test.npy", y_5_test)

## six  individual category
type_of_products = ['TVs', 'cameras','laptops','mobilephone','tablets','video_surveillance']
#for type in type_of_products:
type = 'TVs'
f = open("./ind_category/reviews_"+type+".csv")
lines = f.readlines()
y = np.load("./ind_category/stars_"+type+".npy")
X, y_2_class, y_5_class = get_word_index_and_label(lines, y, 150, vocabulary, word_dict)
train_index = np.random.choice(len(lines),  int(len(lines)*0.8), replace = False)
test_index = np.setdiff1d(range(len(lines)), train_index)
X_train = X[train_index]
y_2_train = y_2_class[train_index]
X_test = X[test_index]
y_2_test = y_2_class[test_index]
np.save("X_2_train_"+type+".npy", X_train)
np.save("y_2_train_"+type+".npy", y_2_train)
np.save("X_2_test_"+type+".npy", X_test)
np.save("y_2_test_"+type+".npy", y_2_test)
y_5_train = y_5_class[train_index]
y_5_test = y_5_class[test_index]
np.save("y_5_train_"+type+".npy", y_5_train)
np.save("y_5_test_"+type+".npy", y_5_test)