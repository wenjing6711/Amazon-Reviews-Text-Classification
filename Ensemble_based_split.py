import numpy as np
import re
import csv
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

## individual category
type_of_products = ['TVs', 'cameras','laptops','mobilephone','tablets','video_surveillance']
begin_length = 10
middle_length = 10
end_length = 10
for type in type_of_products:
    filename =  "./ind_category/reviews_"+type+".csv"
    f = open(filename)
    lines = f.readlines()
    with open("./ind_category/reviews_begin_"+type+".csv", 'w', newline='') as new_f1:
        writer1 = csv.writer(new_f1)
        with open("./ind_category/reviews_middle_" + type + ".csv", 'w', newline='') as new_f2:
            writer2 = csv.writer(new_f2)
            with open("./ind_category/reviews_end_" + type + ".csv", 'w', newline='') as new_f3:
                writer3 = csv.writer(new_f3)

                for line in lines:
                    new_line = clean_str(line)
                    word= np.array(new_line.split())
                    if len(word) > 40:
                        begin_line = ' '.join(word[0:begin_length])
                        #middle_index = np.random.choice(len(word)-1-10, middle_length, replace =False)
                        middle_line = ' '.join(word[15:25])
                        end_line = ' '.join(word[len(word)-end_length:(len(word)-1)])
                    elif 10 <len(word)<=40:
                        begin_line = ' '.join(word[0:begin_length])
                        #middle_index = np.random.choice(len(word)-1, middle_length,replace =False)
                        middle_line = ' '.join(word[10:20])
                        end_line = ' '.join(word[len(word)-end_length:(len(word)-1)])
                    else:
                        begin_line = new_line
                        middle_line = new_line
                        end_line = new_line
                    writer1.writerow([begin_line])
                    writer2.writerow([middle_line])
                    writer3.writerow([end_line])

## single category
begin_length = 10
middle_length = 10
end_length = 10
filename =  "./single_category/reviews.csv"
f = open(filename)
lines = f.readlines()
with open("./single_category/reviews_begin.csv", 'w', newline='') as new_f1:
    writer1 = csv.writer(new_f1)
    with open("./single_category/reviews_middle.csv", 'w', newline='') as new_f2:
        writer2 = csv.writer(new_f2)
        with open("./single_category/reviews_end.csv", 'w', newline='') as new_f3:
            writer3 = csv.writer(new_f3)

            for line in lines:
                new_line = clean_str(line)
                word= np.array(new_line.split())
                if len(word) > 40:
                    begin_line = ' '.join(word[0:begin_length])
                    #middle_index = np.random.choice(10,len(word)-1-10, middle_length, replace=False)
                    middle_line = ' '.join(word[15:25])
                    end_line = ' '.join(word[len(word)-end_length:(len(word)-1)])
                elif 10 <len(word)<=40:
                    begin_line = ' '.join(word[0:begin_length])
                    #middle_index = np.random.choice(0,len(word)-1, middle_length, raplce=False)
                    middle_line = ' '.join(word[10:20])
                    end_line = ' '.join(word[len(word)-end_length:(len(word)-1)])
                else:
                    begin_line = new_line
                    middle_line = new_line
                    end_line = new_line
                writer1.writerow([begin_line])
                writer2.writerow([middle_line])
                writer3.writerow([end_line])
