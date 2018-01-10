import numpy as np
import csv
import re

def clean_str(string):
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\n", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    return string.lower()
# offset =np.random.choice(208052,100000)
offset = np.load("offset.npy")
with open("review_input.csv", 'w', newline='') as f1:
    writer=csv.writer(f1)
    f2 = open('review.csv')
    lines= f2.readlines()
    for i in offset:
        random_line = lines[i]
        new_line = clean_str(random_line)
        writer.writerow([random_line])

with open("titles_input.csv",'w', newline = '') as f3:
    writer = csv.writer(f3)
    f4 = open('titles.csv')
    lines = f4.readlines()
    for i in offset:
        random_line = lines[i]
        new_line = clean_str(random_line)
        writer.writerow([new_line])

star = np.load('stars.npy')
star_input = star[offset]
np.save("star_input", star_input)
