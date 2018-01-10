import json
import numpy as np
import os
import csv
import re
from nltk.corpus import stopwords
import pandas as pd

cachedStopWords = stopwords.words("english")
## remove stop word
def remove_stop_words(text):
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    return text
## remove symbols
def clean_str(string):
    string = re.sub(r"\'s", "is", string)
    string = re.sub(r"\'ve", "have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", "will", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    return string.lower()

type_of_products = ['TVs', 'cameras','laptops','mobilephone','tablets','video_surveillance']
# # generate random_index
# random_index = {}
# for type in type_of_products:
#     foldername = "./AmazonReviews/" + type + '/'
#     list_of_products = os.listdir(foldername)
#     if 400 < len(list_of_products) < 7000:
#         s = np.random.choice(len(list_of_products), int(len(list_of_products)*0.2), replace = False)
#         random_index[type] = list(np.array(list_of_products)[s])
#     elif len(list_of_products) > 7000:
#         s = np.random.choice(len(list_of_products), int(len(list_of_products)*0.1), replace = False)
#         random_index[type] = list(np.array(list_of_products)[s])
#     elif len(list_of_products) < 2000:
#         s= np.random.choice(len(list_of_products), int(len(list_of_products)*0.8), replace = False)
#         random_index[type] = list(np.array(list_of_products)[s])
#     else:
#         s = np.random.choice(len(list_of_products), int(len(list_of_products)*0.4), replace = False)
#         random_index[type] = list(np.array(list_of_products)[s])
#
# with open('ind_random_index', 'w') as outfile:
#     json.dump(random_index, outfile)
random_index = json.load(open('random_index'))
### generate single category data
with open("./single_category/titles.csv", 'w', newline='') as f:
    writer=csv.writer(f)
    for type in type_of_products:
        foldername = "./AmazonReviews/"+type+'/'
        list_of_products = random_index[type]
        for prod in list_of_products:
            filename = foldername + prod
            stuff = json.load(open(filename))['Reviews']
            for review in stuff:
                if review['Title'] != None and review['Content'] != None and review['Overall'] != 'None':
                    new_line = clean_str(review['Title'])
                    new_line = remove_stop_words(new_line)
                    writer.writerow([new_line])

with open("./single_category/review.csv", 'w', newline='') as f:
    writer=csv.writer(f)
    for type in type_of_products:
        foldername = "./AmazonReviews/"+type+'/'
        list_of_products = random_index[type]
        for prod in list_of_products:
            filename = foldername + prod
            stuff = json.load(open(filename))['Reviews']
            for review in stuff:
                if review['Title'] != None and review['Content'] != None and review['Overall'] != 'None':
                    new_line = clean_str(review['Content'])
                    new_line = remove_stop_words(new_line)
                    writer.writerow([new_line])


star = []
for type in type_of_products:
    foldername = "./AmazonReviews/"+type+'/'
    list_of_products = random_index[type]
    for prod in list_of_products:
        filename = foldername + prod
        stuff = json.load(open(filename))['Reviews']
        for review in stuff:
            if review['Title'] != None and review['Content'] != None and review['Overall'] != 'None':
                star.append(int(float(review['Overall'])))
np.save("./single_category/stars.npy", np.array(star))

## generate csv for each category
type_of_products = ['TVs', 'cameras','laptops','mobilephone','tablets','video_surveillance']
ind_random_index = json.load(open('ind_random_index'))
for type in type_of_products:
    save_filename= "./ind_category/titles_"+type+".csv"
    with open(save_filename, 'w', newline='') as f:
        writer=csv.writer(f)
        foldername = "./AmazonReviews/"+type+'/'
        list_of_products = ind_random_index[type]
        for prod in list_of_products:
            filename = foldername + prod
            stuff = json.load(open(filename))['Reviews']
            for review in stuff:
                if review['Title'] != None and review['Content'] != None and review['Overall'] != 'None':
                    new_line = clean_str(review['Title'])
                    new_line = remove_stop_words(new_line)
                    writer.writerow([new_line])

for type in type_of_products:
    save_filename= "./ind_category/reviews_"+type+".csv"
    with open(save_filename, 'w', newline='') as f:
        writer=csv.writer(f)
        foldername = "./AmazonReviews/"+type+'/'
        list_of_products = ind_random_index[type]
        for prod in list_of_products:
            filename = foldername + prod
            stuff = json.load(open(filename))['Reviews']
            for review in stuff:
                if review['Title'] != None and review['Content'] != None and review['Overall'] != 'None':
                    new_line = clean_str(review['Content'])
                    new_line = remove_stop_words(new_line)
                    writer.writerow([new_line])


for type in type_of_products:
    star = []
    foldername = "./AmazonReviews/"+type+'/'
    list_of_products = ind_random_index[type]
    for prod in list_of_products:
        filename = foldername + prod
        stuff = json.load(open(filename))['Reviews']
        for review in stuff:
            if review['Title'] != None and review['Content'] != None and review['Overall'] != 'None':
                star.append(int(float(review['Overall'])))
    save_filename = "./ind_category/stars_" + type + ".csv"
    np.save(save_filename, np.array(star))

for type in type_of_products:
    star = []
    foldername = "./AmazonReviews/" + type + '/'
    list_of_products = os.listdir(foldername)
    for prod in list_of_products:
        filename = foldername + prod
        stuff = json.load(open(filename))['Reviews']
        for review in stuff:
            if review['Overall'] != 'None':
                star.append(int(float(review['Overall'])))
    save_filename = "./ind_category/stars_" + type + ".csv"
    star_pd= pd.DataFrame(star)
    star_pd.to_csv(save_filename)
