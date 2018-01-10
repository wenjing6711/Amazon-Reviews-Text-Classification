from gensim.models import Doc2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import neural_network as nn
from sklearn.ensemble import GradientBoostingClassifier
from collections import Counter
num_of_var = 100

categories = ['cameras','laptops','mobilephone','tablets','TVs','video_surveillance','all' ]

for category in categories:
    print('#'*(len(category)+2))
    print('#' + category + '#')
    print('#' * (len(category) + 2))
    if category != 'all':
        x_title = np.load('./ind_category/doc_topic_titles_' + category + '.npy')
        x_begin = np.load('./ind_category/doc_topic_reviews_begin_' + category + '.npy')
        x_middle = np.load('./ind_category/doc_topic_reviews_middle_' + category + '.npy')
        x_end = np.load('./ind_category/doc_topic_reviews_end_' + category + '.npy')
        y = np.load('./ind_category/doc_topic_titles_y_'+ category + '.npy')

        n = y.shape[0]
        train_idx = np.random.choice(n, int(n * 0.8), replace=False)
        test_idx = np.setdiff1d(list(range(n)), train_idx)

        x_title_train = x_title[train_idx,]
        x_begin_train = x_begin[train_idx,]
        x_middle_train = x_middle[train_idx,]
        x_end_train = x_end[train_idx,]
        y_train = np.asarray(y)[train_idx]

        x_title_test = x_title[test_idx,]
        x_begin_test = x_begin[test_idx,]
        x_middle_test = x_middle[test_idx,]
        x_end_test = x_end[test_idx,]
        y_test = np.asarray(y)[test_idx]
    else:
        x_title = np.load('./single_category/doc_topic_titles.npy')
        x_begin = np.load('./single_category/doc_topic_reviews_begin.npy')
        x_middle = np.load('./single_category/doc_topic_reviews_middle.npy')
        x_end = np.load('./single_category/doc_topic_reviews_end.npy')
        y = np.load('./single_category/doc_topic_titles_y.npy')
        n = y.shape[0]
        train_idx = np.random.choice(n, int(n * 0.8), replace=False)
        test_idx = np.setdiff1d(list(range(n)), train_idx)

        x_title_train = x_title[train_idx,]
        x_begin_train = x_begin[train_idx,]
        x_middle_train = x_middle[train_idx,]
        x_end_train = x_end[train_idx,]
        y_train = np.asarray(y)[train_idx]

        x_title_test = x_title[test_idx,]
        x_begin_test = x_begin[test_idx,]
        x_middle_test = x_middle[test_idx,]
        x_end_test = x_end[test_idx,]
        y_test = np.asarray(y)[test_idx]


    #five class:
    # Logistic regression
    #title:
    lr_title = LogisticRegression()
    lr_title.fit(x_title_train, np.ravel(y_train))
    preds_lr_title = lr_title.predict(x_title_train)
    train_y_array = np.array(y_train)
    pred_train_lr_title = sum(np.equal(train_y_array, preds_lr_title)) / len(y_train)
    print("5 class, title for " + category + " lr model training acc: " + str(pred_train_lr_title)) # 0.87837403599 0.611819837189
    preds_lr_test_title = lr_title.predict(x_title_test)
    test_y_array = np.array(y_test)
    pred_test_lr_title = sum(np.equal(test_y_array, preds_lr_test_title)) / len(y_test)
    print("5 class, title for " + category + " lr model testing acc: " + str(pred_test_lr_title))  # 0.876052355348 0.610998050599
    #begin
    lr_begin = LogisticRegression()
    lr_begin.fit(x_begin_train, np.ravel(y_train))
    preds_lr_begin = lr_begin.predict(x_begin_train)
    train_y_array = np.array(y_train)
    pred_train_lr_begin = sum(np.equal(train_y_array, preds_lr_begin)) / len(y_train)
    print("5 class, begin for " + category + " lr model training acc: " + str(
        pred_train_lr_begin))  # 0.87837403599 0.611819837189
    preds_lr_test_begin = lr_begin.predict(x_begin_test)
    test_y_array = np.array(y_test)
    pred_test_lr_begin = sum(np.equal(test_y_array, preds_lr_test_begin)) / len(y_test)
    print("5 class, begin for " + category + " lr model testing acc: " + str(pred_test_lr_begin))
    #middle
    lr_middle = LogisticRegression()
    lr_middle.fit(x_middle_train, np.ravel(y_train))
    preds_lr_middle = lr_middle.predict(x_middle_train)
    train_y_array = np.array(y_train)
    pred_train_lr_middle = sum(np.equal(train_y_array, preds_lr_middle)) / len(y_train)
    print("5 class, middle for " + category + " lr model training acc: " + str(
        pred_train_lr_middle))  # 0.87837403599 0.611819837189
    preds_lr_test_middle = lr_middle.predict(x_middle_test)
    test_y_array = np.array(y_test)
    pred_test_lr_middle = sum(np.equal(test_y_array, preds_lr_test_middle)) / len(y_test)
    print("5 class, middle for " + category + " lr model testing acc: " + str(pred_test_lr_middle))
    #end
    lr_end = LogisticRegression()
    lr_end.fit(x_end_train, np.ravel(y_train))
    preds_lr_end = lr_end.predict(x_end_train)
    train_y_array = np.array(y_train)
    pred_train_lr_end = sum(np.equal(train_y_array, preds_lr_end)) / len(y_train)
    print("5 class, end for " + category + " lr model training acc: " + str(
        pred_train_lr_end))  # 0.87837403599 0.611819837189
    preds_lr_test_end = lr_end.predict(x_end_test)
    test_y_array = np.array(y_test)
    pred_test_lr_end = sum(np.equal(test_y_array, preds_lr_test_end)) / len(y_test)
    print("5 class, end for " + category + " lr model testing acc: " + str(pred_test_lr_end))
    #mv
    pred_lr_train_mv = [Counter([preds_lr_title[i], preds_lr_begin[i],
                                         preds_lr_middle[i], preds_lr_end[i], ]).most_common(1)[0][0]
                        for i in range(len(y_train))]
    pred_lr_train_acc_mv = sum(np.equal(train_y_array, pred_lr_train_mv)) / len(y_train)
    print("5 class, mv for " + category + " lr model training acc: " + str(
        pred_lr_train_acc_mv))
    pred_lr_test_mv = [Counter([preds_lr_test_title[i],preds_lr_test_begin[i],
                                   preds_lr_test_middle[i],preds_lr_test_end[i],]).most_common(1)[0][0]
                  for i in range(len(y_test))]
    pred_lr_test_acc_mv = sum(np.equal(test_y_array, pred_lr_test_mv)) / len(y_test)
    print("5 class, mv for " + category + " lr model testing acc: " + str(
        pred_lr_test_acc_mv))

    #random forest
    #tittle
    print('#'*50)
    print('#'*50)
    rfc_title = RandomForestClassifier(n_estimators=200, max_depth=8)
    rfc_title.fit(x_title_train, np.ravel(y_train))
    preds_rfc_title = rfc_title.predict(x_title_train)
    train_y_array = np.array(y_train)
    pred_train_title = sum(np.equal(train_y_array, preds_rfc_title)) / len(y_train)
    print("5 class, title for " + category + " rfc model training acc: " + str(
        pred_train_title))  # 0.87837403599 0.611819837189
    preds_rfc_test_title = rfc_title.predict(x_title_test)
    test_y_array = np.array(y_test)
    pred_test_title = sum(np.equal(test_y_array, preds_rfc_test_title)) / len(y_test)
    print("5 class, title for " + category + " rfc model testing acc: " + str(
        pred_test_title))  # 0.876052355348 0.610998050599
    # begin
    rfc_begin = RandomForestClassifier(n_estimators=200, max_depth=8)
    rfc_begin.fit(x_begin_train, np.ravel(y_train))
    preds_rfc_begin = rfc_begin.predict(x_begin_train)
    train_y_array = np.array(y_train)
    pred_train_begin = sum(np.equal(train_y_array, preds_rfc_begin)) / len(y_train)
    print("5 class, begin for " + category + " rfc model training acc: " + str(
        pred_train_begin))  # 0.87837403599 0.611819837189
    preds_rfc_test_begin = rfc_begin.predict(x_begin_test)
    test_y_array = np.array(y_test)
    pred_test_begin = sum(np.equal(test_y_array, preds_rfc_test_begin)) / len(y_test)
    print("5 class, begin for " + category + " rfc model testing acc: " + str(
        pred_test_begin))  # 0.876052355348 0.610998050599
    # middle
    rfc_middle = RandomForestClassifier(n_estimators=200, max_depth=8)
    rfc_middle.fit(x_middle_train, np.ravel(y_train))
    preds_rfc_middle = rfc_middle.predict(x_middle_train)
    train_y_array = np.array(y_train)
    pred_train_middle = sum(np.equal(train_y_array, preds_rfc_middle)) / len(y_train)
    print("5 class, middle for " + category + " rfc model training acc: " + str(
        pred_train_middle))  # 0.87837403599 0.611819837189
    preds_rfc_test_middle = rfc_middle.predict(x_middle_test)
    test_y_array = np.array(y_test)
    pred_test_middle = sum(np.equal(test_y_array, preds_rfc_test_middle)) / len(y_test)
    print("5 class, middle for " + category + " rfc model testing acc: " + str(
        pred_test_middle))  # 0.876052355348 0.610998050599
    # end
    rfc_end = RandomForestClassifier(n_estimators=200, max_depth=8)
    rfc_end.fit(x_end_train, np.ravel(y_train))
    preds_rfc_end = rfc_end.predict(x_end_train)
    train_y_array = np.array(y_train)
    pred_train_end = sum(np.equal(train_y_array, preds_rfc_end)) / len(y_train)
    print("5 class, end for " + category + " rfc model training acc: " + str(
        pred_train_end))  # 0.87837403599 0.611819837189
    preds_rfc_test_end = rfc_end.predict(x_end_test)
    test_y_array = np.array(y_test)
    pred_test_end = sum(np.equal(test_y_array, preds_rfc_test_end)) / len(y_test)
    print("5 class, end for " + category + " rfc model testing acc: " + str(
        pred_test_end))  # 0.876052355348 0.610998050599
    # mv
    pred_rfc_train_mv = [Counter([preds_rfc_title[i], preds_rfc_begin[i],
                                  preds_rfc_middle[i], preds_rfc_end[i], ]).most_common(1)[0][0]
                         for i in range(len(y_train))]
    pred_rfc_train_acc_mv = sum(np.equal(train_y_array, pred_rfc_train_mv)) / len(y_train)
    print("5 class, mv for " + category + " rfc model training acc: " + str(
        pred_rfc_train_acc_mv))
    pred_rfc_test_mv = [Counter([preds_rfc_test_title[i], preds_rfc_test_begin[i],
                                 preds_rfc_test_middle[i], preds_rfc_test_end[i], ]).most_common(1)[0][0]
                        for i in range(len(y_test))]
    pred_rfc_test_acc_mv = sum(np.equal(test_y_array, pred_rfc_test_mv)) / len(y_test)
    print("5 class, mv for " + category + " rfc model testing acc: " + str(
        pred_rfc_test_acc_mv))

    ## NN
    #title
    print('#'*50)
    print('#'*50)
    nnc_title = nn.MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 30, 30))
    nnc_title.fit(x_title_train, np.ravel(y_train))
    preds_nnc_title = nnc_title.predict(x_title_train)
    train_y_array = np.array(y_train)
    pred_train_title = sum(np.equal(train_y_array, preds_nnc_title)) / len(y_train)
    print("5 class, title for " + category + " nnc model training acc: " + str(
        pred_train_title))  # 0.87837403599 0.611819837189
    preds_nnc_test_title = nnc_title.predict(x_title_test)
    test_y_array = np.array(y_test)
    pred_test_title = sum(np.equal(test_y_array, preds_nnc_test_title)) / len(y_test)
    print("5 class, title for " + category + " nnc model testing acc: " + str(
        pred_test_title))  # 0.876052355348 0.610998050599
    # begin
    nnc_begin = nn.MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 30, 30))
    nnc_begin.fit(x_begin_train, np.ravel(y_train))
    preds_nnc_begin = nnc_begin.predict(x_begin_train)
    train_y_array = np.array(y_train)
    pred_train_begin = sum(np.equal(train_y_array, preds_nnc_begin)) / len(y_train)
    print("5 class, begin for " + category + " nnc model training acc: " + str(
        pred_train_begin))  # 0.87837403599 0.611819837189
    preds_nnc_test_begin = nnc_begin.predict(x_begin_test)
    test_y_array = np.array(y_test)
    pred_test_begin = sum(np.equal(test_y_array, preds_nnc_test_begin)) / len(y_test)
    print("5 class, begin for " + category + " nnc model testing acc: " + str(
        pred_test_begin))  # 0.876052355348 0.610998050599
    # middle
    nnc_middle = nn.MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 30, 30))
    nnc_middle.fit(x_middle_train, np.ravel(y_train))
    preds_nnc_middle = nnc_middle.predict(x_middle_train)
    train_y_array = np.array(y_train)
    pred_train_middle = sum(np.equal(train_y_array, preds_nnc_middle)) / len(y_train)
    print("5 class, middle for " + category + " nnc model training acc: " + str(
        pred_train_middle))  # 0.87837403599 0.611819837189
    preds_nnc_test_middle = nnc_middle.predict(x_middle_test)
    test_y_array = np.array(y_test)
    pred_test_middle = sum(np.equal(test_y_array, preds_nnc_test_middle)) / len(y_test)
    print("5 class, middle for " + category + " nnc model testing acc: " + str(
        pred_test_middle))  # 0.876052355348 0.610998050599
    # end
    nnc_end = nn.MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 30, 30))
    nnc_end.fit(x_end_train, np.ravel(y_train))
    preds_nnc_end = nnc_end.predict(x_end_train)
    train_y_array = np.array(y_train)
    pred_train_end = sum(np.equal(train_y_array, preds_nnc_end)) / len(y_train)
    print("5 class, end for " + category + " nnc model training acc: " + str(
        pred_train_end))  # 0.87837403599 0.611819837189
    preds_nnc_test_end = nnc_end.predict(x_end_test)
    test_y_array = np.array(y_test)
    pred_test_end = sum(np.equal(test_y_array, preds_nnc_test_end)) / len(y_test)
    print("5 class, end for " + category + " nnc model testing acc: " + str(
        pred_test_end))  # 0.876052355348 0.610998050599
    # mv
    pred_nnc_train_mv = [Counter([preds_nnc_title[i], preds_nnc_begin[i],
                                  preds_nnc_middle[i], preds_nnc_end[i], ]).most_common(1)[0][0]
                         for i in range(len(y_train))]
    pred_nnc_train_acc_mv = sum(np.equal(train_y_array, pred_nnc_train_mv)) / len(y_train)
    print("5 class, mv for " + category + " nnc model training acc: " + str(
        pred_nnc_train_acc_mv))
    pred_nnc_test_mv = [Counter([preds_nnc_test_title[i], preds_nnc_test_begin[i],
                                 preds_nnc_test_middle[i], preds_nnc_test_end[i], ]).most_common(1)[0][0]
                        for i in range(len(y_test))]
    pred_nnc_test_acc_mv = sum(np.equal(test_y_array, pred_nnc_test_mv)) / len(y_test)
    print("5 class, mv for " + category + " nnc model testing acc: " + str(
        pred_nnc_test_acc_mv))

    #XGB
    #title
    print('#'*50)
    print('#'*50)
    gbc_title = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.05)
    gbc_title.fit(x_title_train, np.ravel(y_train))
    preds_gbc_title = gbc_title.predict(x_title_train)
    train_y_array = np.array(y_train)
    pred_train_title = sum(np.equal(train_y_array, preds_gbc_title)) / len(y_train)
    print("5 class, title for " + category + " gbc model training acc: " + str(
        pred_train_title))  # 0.87837403599 0.611819837189
    preds_gbc_test_title = gbc_title.predict(x_title_test)
    test_y_array = np.array(y_test)
    pred_test_title = sum(np.equal(test_y_array, preds_gbc_test_title)) / len(y_test)
    print("5 class, title for " + category + " gbc model testing acc: " + str(
        pred_test_title))  # 0.876052355348 0.610998050599
    # begin
    gbc_begin = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.05)
    gbc_begin.fit(x_begin_train, np.ravel(y_train))
    preds_gbc_begin = gbc_begin.predict(x_begin_train)
    train_y_array = np.array(y_train)
    pred_train_begin = sum(np.equal(train_y_array, preds_gbc_begin)) / len(y_train)
    print("5 class, begin for " + category + " gbc model training acc: " + str(
        pred_train_begin))  # 0.87837403599 0.611819837189
    preds_gbc_test_begin = gbc_begin.predict(x_begin_test)
    test_y_array = np.array(y_test)
    pred_test_begin = sum(np.equal(test_y_array, preds_gbc_test_begin)) / len(y_test)
    print("5 class, begin for " + category + " gbc model testing acc: " + str(
        pred_test_begin))  # 0.876052355348 0.610998050599
    # middle
    gbc_middle = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.05)
    gbc_middle.fit(x_middle_train, np.ravel(y_train))
    preds_gbc_middle = gbc_middle.predict(x_middle_train)
    train_y_array = np.array(y_train)
    pred_train_middle = sum(np.equal(train_y_array, preds_gbc_middle)) / len(y_train)
    print("5 class, middle for " + category + " gbc model training acc: " + str(
        pred_train_middle))  # 0.87837403599 0.611819837189
    preds_gbc_test_middle = gbc_middle.predict(x_middle_test)
    test_y_array = np.array(y_test)
    pred_test_middle = sum(np.equal(test_y_array, preds_gbc_test_middle)) / len(y_test)
    print("5 class, middle for " + category + " gbc model testing acc: " + str(
        pred_test_middle))  # 0.876052355348 0.610998050599
    # end
    gbc_end = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.05)
    gbc_end.fit(x_end_train, np.ravel(y_train))
    preds_gbc_end = gbc_end.predict(x_end_train)
    train_y_array = np.array(y_train)
    pred_train_end = sum(np.equal(train_y_array, preds_gbc_end)) / len(y_train)
    print("5 class, end for " + category + " gbc model training acc: " + str(
        pred_train_end))  # 0.87837403599 0.611819837189
    preds_gbc_test_end = gbc_end.predict(x_end_test)
    test_y_array = np.array(y_test)
    pred_test_end = sum(np.equal(test_y_array, preds_gbc_test_end)) / len(y_test)
    print("5 class, end for " + category + " gbc model testing acc: " + str(
        pred_test_end))  # 0.876052355348 0.610998050599
    # mv
    pred_gbc_train_mv = [Counter([preds_gbc_title[i], preds_gbc_begin[i],
                                  preds_gbc_middle[i], preds_gbc_end[i], ]).most_common(1)[0][0]
                         for i in range(len(y_train))]
    pred_gbc_train_acc_mv = sum(np.equal(train_y_array, pred_gbc_train_mv)) / len(y_train)
    print("5 class, mv for " + category + " gbc model training acc: " + str(
        pred_gbc_train_acc_mv))
    pred_gbc_test_mv = [Counter([preds_gbc_test_title[i], preds_gbc_test_begin[i],
                                 preds_gbc_test_middle[i], preds_gbc_test_end[i], ]).most_common(1)[0][0]
                        for i in range(len(y_test))]
    pred_gbc_test_acc_mv = sum(np.equal(test_y_array, pred_gbc_test_mv)) / len(y_test)
    print("5 class, mv for " + category + " gbc model testing acc: " + str(
        pred_gbc_test_acc_mv))

    #Combined model:
    print('#'*50)
    print('#'*50)
    pred_all_train_mv = [Counter([preds_lr_title[i], preds_lr_begin[i],preds_lr_middle[i], preds_lr_end[i],
                                  preds_rfc_title[i], preds_rfc_begin[i], preds_rfc_middle[i], preds_rfc_end[i],
                                  preds_nnc_title[i], preds_nnc_begin[i], preds_nnc_middle[i], preds_nnc_end[i],
                                  preds_gbc_title[i], preds_gbc_begin[i],
                                  preds_gbc_middle[i], preds_gbc_end[i], ]).most_common(1)[0][0]
                         for i in range(len(y_train))]
    pred_all_train_acc_mv = sum(np.equal(train_y_array, pred_all_train_mv)) / len(y_train)
    print("5 class, mv for " + category + " all model training acc: " + str(
        pred_all_train_acc_mv))
    pred_all_test_mv = [Counter([preds_lr_test_title[i], preds_lr_test_begin[i],preds_lr_test_middle[i], preds_lr_test_end[i],
                                 preds_rfc_test_title[i], preds_rfc_test_begin[i], preds_rfc_test_middle[i],
                                 preds_rfc_test_end[i],preds_nnc_test_title[i], preds_nnc_test_begin[i],preds_nnc_test_middle[i],
                                 preds_nnc_test_end[i],preds_gbc_test_title[i], preds_gbc_test_begin[i],
                                 preds_gbc_test_middle[i], preds_gbc_test_end[i], ]).most_common(1)[0][0]
                        for i in range(len(y_test))]
    pred_all_test_acc_mv = sum(np.equal(test_y_array, pred_all_test_mv)) / len(y_test)
    print("5 class, mv for " + category + " all model testing acc: " + str(
        pred_all_test_acc_mv))
    print('#' * 50)
    print('#' * 50)

    y_train = [0 if i < 3 else 1 for i in y_train]
    y_test = [0 if i < 3 else 1 for i in y_test]

    # two class:
    # Logistic regression
    # title:
    lr_title = LogisticRegression()
    lr_title.fit(x_title_train, np.ravel(y_train))
    preds_lr_title = lr_title.predict(x_title_train)
    train_y_array = np.array(y_train)
    pred_train_lr_title = sum(np.equal(train_y_array, preds_lr_title)) / len(y_train)
    print("2 class, title for " + category + " lr model training acc: " + str(
        pred_train_lr_title))  # 0.87837403599 0.611819837189
    preds_lr_test_title = lr_title.predict(x_title_test)
    test_y_array = np.array(y_test)
    pred_test_lr_title = sum(np.equal(test_y_array, preds_lr_test_title)) / len(y_test)
    print("2 class, title for " + category + " lr model testing acc: " + str(
        pred_test_lr_title))  # 0.876052355348 0.610998050599
    # begin
    lr_begin = LogisticRegression()
    lr_begin.fit(x_begin_train, np.ravel(y_train))
    preds_lr_begin = lr_begin.predict(x_begin_train)
    train_y_array = np.array(y_train)
    pred_train_lr_begin = sum(np.equal(train_y_array, preds_lr_begin)) / len(y_train)
    print("2 class, begin for " + category + " lr model training acc: " + str(
        pred_train_lr_begin))  # 0.87837403599 0.611819837189
    preds_lr_test_begin = lr_begin.predict(x_begin_test)
    test_y_array = np.array(y_test)
    pred_test_lr_begin = sum(np.equal(test_y_array, preds_lr_test_begin)) / len(y_test)
    print("2 class, begin for " + category + " lr model testing acc: " + str(pred_test_lr_begin))
    # middle
    lr_middle = LogisticRegression()
    lr_middle.fit(x_middle_train, np.ravel(y_train))
    preds_lr_middle = lr_middle.predict(x_middle_train)
    train_y_array = np.array(y_train)
    pred_train_lr_middle = sum(np.equal(train_y_array, preds_lr_middle)) / len(y_train)
    print("2 class, middle for " + category + " lr model training acc: " + str(
        pred_train_lr_middle))  # 0.87837403599 0.611819837189
    preds_lr_test_middle = lr_middle.predict(x_middle_test)
    test_y_array = np.array(y_test)
    pred_test_lr_middle = sum(np.equal(test_y_array, preds_lr_test_middle)) / len(y_test)
    print("2 class, middle for " + category + " lr model testing acc: " + str(pred_test_lr_middle))
    # end
    lr_end = LogisticRegression()
    lr_end.fit(x_end_train, np.ravel(y_train))
    preds_lr_end = lr_end.predict(x_end_train)
    train_y_array = np.array(y_train)
    pred_train_lr_end = sum(np.equal(train_y_array, preds_lr_end)) / len(y_train)
    print("2 class, end for " + category + " lr model training acc: " + str(
        pred_train_lr_end))  # 0.87837403599 0.611819837189
    preds_lr_test_end = lr_end.predict(x_end_test)
    test_y_array = np.array(y_test)
    pred_test_lr_end = sum(np.equal(test_y_array, preds_lr_test_end)) / len(y_test)
    print("2 class, end for " + category + " lr model testing acc: " + str(pred_test_lr_end))
    # mv
    pred_lr_train_mv = [Counter([preds_lr_title[i], preds_lr_begin[i],
                                 preds_lr_middle[i], preds_lr_end[i], ]).most_common(1)[0][0]
                        for i in range(len(y_train))]
    pred_lr_train_acc_mv = sum(np.equal(train_y_array, pred_lr_train_mv)) / len(y_train)
    print("2 class, mv for " + category + " lr model training acc: " + str(
        pred_lr_train_acc_mv))
    pred_lr_test_mv = [Counter([preds_lr_test_title[i], preds_lr_test_begin[i],
                                preds_lr_test_middle[i], preds_lr_test_end[i], ]).most_common(1)[0][0]
                       for i in range(len(y_test))]
    pred_lr_test_acc_mv = sum(np.equal(test_y_array, pred_lr_test_mv)) / len(y_test)
    print("2 class, mv for " + category + " lr model testing acc: " + str(
        pred_lr_test_acc_mv))

    # random forest
    # tittle
    print('#' * 50)
    print('#' * 50)
    rfc_title = RandomForestClassifier(n_estimators=200, max_depth=8)
    rfc_title.fit(x_title_train, np.ravel(y_train))
    preds_rfc_title = rfc_title.predict(x_title_train)
    train_y_array = np.array(y_train)
    pred_train_title = sum(np.equal(train_y_array, preds_rfc_title)) / len(y_train)
    print("2 class, title for " + category + " rfc model training acc: " + str(
        pred_train_title))  # 0.87837403599 0.611819837189
    preds_rfc_test_title = rfc_title.predict(x_title_test)
    test_y_array = np.array(y_test)
    pred_test_title = sum(np.equal(test_y_array, preds_rfc_test_title)) / len(y_test)
    print("2 class, title for " + category + " rfc model testing acc: " + str(
        pred_test_title))  # 0.876052355348 0.610998050599
    # begin
    rfc_begin = RandomForestClassifier(n_estimators=200, max_depth=8)
    rfc_begin.fit(x_begin_train, np.ravel(y_train))
    preds_rfc_begin = rfc_begin.predict(x_begin_train)
    train_y_array = np.array(y_train)
    pred_train_begin = sum(np.equal(train_y_array, preds_rfc_begin)) / len(y_train)
    print("2 class, begin for " + category + " rfc model training acc: " + str(
        pred_train_begin))  # 0.87837403599 0.611819837189
    preds_rfc_test_begin = rfc_begin.predict(x_begin_test)
    test_y_array = np.array(y_test)
    pred_test_begin = sum(np.equal(test_y_array, preds_rfc_test_begin)) / len(y_test)
    print("2 class, begin for " + category + " rfc model testing acc: " + str(
        pred_test_begin))  # 0.876052355348 0.610998050599
    # middle
    rfc_middle = RandomForestClassifier(n_estimators=200, max_depth=8)
    rfc_middle.fit(x_middle_train, np.ravel(y_train))
    preds_rfc_middle = rfc_middle.predict(x_middle_train)
    train_y_array = np.array(y_train)
    pred_train_middle = sum(np.equal(train_y_array, preds_rfc_middle)) / len(y_train)
    print("2 class, middle for " + category + " rfc model training acc: " + str(
        pred_train_middle))  # 0.87837403599 0.611819837189
    preds_rfc_test_middle = rfc_middle.predict(x_middle_test)
    test_y_array = np.array(y_test)
    pred_test_middle = sum(np.equal(test_y_array, preds_rfc_test_middle)) / len(y_test)
    print("2 class, middle for " + category + " rfc model testing acc: " + str(
        pred_test_middle))  # 0.876052355348 0.610998050599
    # end
    rfc_end = RandomForestClassifier(n_estimators=200, max_depth=8)
    rfc_end.fit(x_end_train, np.ravel(y_train))
    preds_rfc_end = rfc_end.predict(x_end_train)
    train_y_array = np.array(y_train)
    pred_train_end = sum(np.equal(train_y_array, preds_rfc_end)) / len(y_train)
    print("2 class, end for " + category + " rfc model training acc: " + str(
        pred_train_end))  # 0.87837403599 0.611819837189
    preds_rfc_test_end = rfc_end.predict(x_end_test)
    test_y_array = np.array(y_test)
    pred_test_end = sum(np.equal(test_y_array, preds_rfc_test_end)) / len(y_test)
    print("2 class, end for " + category + " rfc model testing acc: " + str(
        pred_test_end))  # 0.876052355348 0.610998050599
    # mv
    pred_rfc_train_mv = [Counter([preds_rfc_title[i], preds_rfc_begin[i],
                                  preds_rfc_middle[i], preds_rfc_end[i], ]).most_common(1)[0][0]
                         for i in range(len(y_train))]
    pred_rfc_train_acc_mv = sum(np.equal(train_y_array, pred_rfc_train_mv)) / len(y_train)
    print("2 class, mv for " + category + " rfc model training acc: " + str(
        pred_rfc_train_acc_mv))
    pred_rfc_test_mv = [Counter([preds_rfc_test_title[i], preds_rfc_test_begin[i],
                                 preds_rfc_test_middle[i], preds_rfc_test_end[i], ]).most_common(1)[0][0]
                        for i in range(len(y_test))]
    pred_rfc_test_acc_mv = sum(np.equal(test_y_array, pred_rfc_test_mv)) / len(y_test)
    print("2 class, mv for " + category + " rfc model testing acc: " + str(
        pred_rfc_test_acc_mv))

    ## NN
    # title
    print('#' * 50)
    print('#' * 50)
    nnc_title = nn.MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 30, 30))
    nnc_title.fit(x_title_train, np.ravel(y_train))
    preds_nnc_title = nnc_title.predict(x_title_train)
    train_y_array = np.array(y_train)
    pred_train_title = sum(np.equal(train_y_array, preds_nnc_title)) / len(y_train)
    print("2 class, title for " + category + " nnc model training acc: " + str(
        pred_train_title))  # 0.87837403599 0.611819837189
    preds_nnc_test_title = nnc_title.predict(x_title_test)
    test_y_array = np.array(y_test)
    pred_test_title = sum(np.equal(test_y_array, preds_nnc_test_title)) / len(y_test)
    print("2 class, title for " + category + " nnc model testing acc: " + str(
        pred_test_title))  # 0.876052355348 0.610998050599
    # begin
    nnc_begin = nn.MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 30, 30))
    nnc_begin.fit(x_begin_train, np.ravel(y_train))
    preds_nnc_begin = nnc_begin.predict(x_begin_train)
    train_y_array = np.array(y_train)
    pred_train_begin = sum(np.equal(train_y_array, preds_nnc_begin)) / len(y_train)
    print("2 class, begin for " + category + " nnc model training acc: " + str(
        pred_train_begin))  # 0.87837403599 0.611819837189
    preds_nnc_test_begin = nnc_begin.predict(x_begin_test)
    test_y_array = np.array(y_test)
    pred_test_begin = sum(np.equal(test_y_array, preds_nnc_test_begin)) / len(y_test)
    print("2 class, begin for " + category + " nnc model testing acc: " + str(
        pred_test_begin))  # 0.876052355348 0.610998050599
    # middle
    nnc_middle = nn.MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 30, 30))
    nnc_middle.fit(x_middle_train, np.ravel(y_train))
    preds_nnc_middle = nnc_middle.predict(x_middle_train)
    train_y_array = np.array(y_train)
    pred_train_middle = sum(np.equal(train_y_array, preds_nnc_middle)) / len(y_train)
    print("2 class, middle for " + category + " nnc model training acc: " + str(
        pred_train_middle))  # 0.87837403599 0.611819837189
    preds_nnc_test_middle = nnc_middle.predict(x_middle_test)
    test_y_array = np.array(y_test)
    pred_test_middle = sum(np.equal(test_y_array, preds_nnc_test_middle)) / len(y_test)
    print("2 class, middle for " + category + " nnc model testing acc: " + str(
        pred_test_middle))  # 0.876052355348 0.610998050599
    # end
    nnc_end = nn.MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 30, 30))
    nnc_end.fit(x_end_train, np.ravel(y_train))
    preds_nnc_end = nnc_end.predict(x_end_train)
    train_y_array = np.array(y_train)
    pred_train_end = sum(np.equal(train_y_array, preds_nnc_end)) / len(y_train)
    print("2 class, end for " + category + " nnc model training acc: " + str(
        pred_train_end))  # 0.87837403599 0.611819837189
    preds_nnc_test_end = nnc_end.predict(x_end_test)
    test_y_array = np.array(y_test)
    pred_test_end = sum(np.equal(test_y_array, preds_nnc_test_end)) / len(y_test)
    print("2 class, end for " + category + " nnc model testing acc: " + str(
        pred_test_end))  # 0.876052355348 0.610998050599
    # mv
    pred_nnc_train_mv = [Counter([preds_nnc_title[i], preds_nnc_begin[i],
                                  preds_nnc_middle[i], preds_nnc_end[i], ]).most_common(1)[0][0]
                         for i in range(len(y_train))]
    pred_nnc_train_acc_mv = sum(np.equal(train_y_array, pred_nnc_train_mv)) / len(y_train)
    print("2 class, mv for " + category + " nnc model training acc: " + str(
        pred_nnc_train_acc_mv))
    pred_nnc_test_mv = [Counter([preds_nnc_test_title[i], preds_nnc_test_begin[i],
                                 preds_nnc_test_middle[i], preds_nnc_test_end[i], ]).most_common(1)[0][0]
                        for i in range(len(y_test))]
    pred_nnc_test_acc_mv = sum(np.equal(test_y_array, pred_nnc_test_mv)) / len(y_test)
    print("2 class, mv for " + category + " nnc model testing acc: " + str(
        pred_nnc_test_acc_mv))

    # XGB
    # title
    print('#' * 50)
    print('#' * 50)
    gbc_title = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.05)
    gbc_title.fit(x_title_train, np.ravel(y_train))
    preds_gbc_title = gbc_title.predict(x_title_train)
    train_y_array = np.array(y_train)
    pred_train_title = sum(np.equal(train_y_array, preds_gbc_title)) / len(y_train)
    print("2 class, title for " + category + " gbc model training acc: " + str(
        pred_train_title))  # 0.87837403599 0.611819837189
    preds_gbc_test_title = gbc_title.predict(x_title_test)
    test_y_array = np.array(y_test)
    pred_test_title = sum(np.equal(test_y_array, preds_gbc_test_title)) / len(y_test)
    print("2 class, title for " + category + " gbc model testing acc: " + str(
        pred_test_title))  # 0.876052355348 0.610998050599
    # begin
    gbc_begin = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.05)
    gbc_begin.fit(x_begin_train, np.ravel(y_train))
    preds_gbc_begin = gbc_begin.predict(x_begin_train)
    train_y_array = np.array(y_train)
    pred_train_begin = sum(np.equal(train_y_array, preds_gbc_begin)) / len(y_train)
    print("2 class, begin for " + category + " gbc model training acc: " + str(
        pred_train_begin))  # 0.87837403599 0.611819837189
    preds_gbc_test_begin = gbc_begin.predict(x_begin_test)
    test_y_array = np.array(y_test)
    pred_test_begin = sum(np.equal(test_y_array, preds_gbc_test_begin)) / len(y_test)
    print("2 class, begin for " + category + " gbc model testing acc: " + str(
        pred_test_begin))  # 0.876052355348 0.610998050599
    # middle
    gbc_middle = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.05)
    gbc_middle.fit(x_middle_train, np.ravel(y_train))
    preds_gbc_middle = gbc_middle.predict(x_middle_train)
    train_y_array = np.array(y_train)
    pred_train_middle = sum(np.equal(train_y_array, preds_gbc_middle)) / len(y_train)
    print("2 class, middle for " + category + " gbc model training acc: " + str(
        pred_train_middle))  # 0.87837403599 0.611819837189
    preds_gbc_test_middle = gbc_middle.predict(x_middle_test)
    test_y_array = np.array(y_test)
    pred_test_middle = sum(np.equal(test_y_array, preds_gbc_test_middle)) / len(y_test)
    print("2 class, middle for " + category + " gbc model testing acc: " + str(
        pred_test_middle))  # 0.876052355348 0.610998050599
    # end
    gbc_end = GradientBoostingClassifier(n_estimators=50, max_depth=5, learning_rate=0.05)
    gbc_end.fit(x_end_train, np.ravel(y_train))
    preds_gbc_end = gbc_end.predict(x_end_train)
    train_y_array = np.array(y_train)
    pred_train_end = sum(np.equal(train_y_array, preds_gbc_end)) / len(y_train)
    print("2 class, end for " + category + " gbc model training acc: " + str(
        pred_train_end))  # 0.87837403599 0.611819837189
    preds_gbc_test_end = gbc_end.predict(x_end_test)
    test_y_array = np.array(y_test)
    pred_test_end = sum(np.equal(test_y_array, preds_gbc_test_end)) / len(y_test)
    print("2 class, end for " + category + " gbc model testing acc: " + str(
        pred_test_end))  # 0.876052355348 0.610998050599
    # mv
    pred_gbc_train_mv = [Counter([preds_gbc_title[i], preds_gbc_begin[i],
                                  preds_gbc_middle[i], preds_gbc_end[i], ]).most_common(1)[0][0]
                         for i in range(len(y_train))]
    pred_gbc_train_acc_mv = sum(np.equal(train_y_array, pred_gbc_train_mv)) / len(y_train)
    print("2 class, mv for " + category + " gbc model training acc: " + str(
        pred_gbc_train_acc_mv))
    pred_gbc_test_mv = [Counter([preds_gbc_test_title[i], preds_gbc_test_begin[i],
                                 preds_gbc_test_middle[i], preds_gbc_test_end[i], ]).most_common(1)[0][0]
                        for i in range(len(y_test))]
    pred_gbc_test_acc_mv = sum(np.equal(test_y_array, pred_gbc_test_mv)) / len(y_test)
    print("2 class, mv for " + category + " gbc model testing acc: " + str(
        pred_gbc_test_acc_mv))

    # Combined model:
    print('#' * 50)
    print('#' * 50)
    pred_all_train_mv = [Counter([preds_lr_title[i], preds_lr_begin[i], preds_lr_middle[i], preds_lr_end[i],
                                  preds_rfc_title[i], preds_rfc_begin[i], preds_rfc_middle[i], preds_rfc_end[i],
                                  preds_nnc_title[i], preds_nnc_begin[i], preds_nnc_middle[i], preds_nnc_end[i],
                                  preds_gbc_title[i], preds_gbc_begin[i],
                                  preds_gbc_middle[i], preds_gbc_end[i], ]).most_common(1)[0][0]
                         for i in range(len(y_train))]
    pred_all_train_acc_mv = sum(np.equal(train_y_array, pred_all_train_mv)) / len(y_train)
    print("2 class, mv for " + category + " all model training acc: " + str(
        pred_all_train_acc_mv))
    pred_all_test_mv = [
        Counter([preds_lr_test_title[i], preds_lr_test_begin[i], preds_lr_test_middle[i], preds_lr_test_end[i],
                 preds_rfc_test_title[i], preds_rfc_test_begin[i], preds_rfc_test_middle[i],
                 preds_rfc_test_end[i], preds_nnc_test_title[i], preds_nnc_test_begin[i], preds_nnc_test_middle[i],
                 preds_nnc_test_end[i], preds_gbc_test_title[i], preds_gbc_test_begin[i],
                 preds_gbc_test_middle[i], preds_gbc_test_end[i], ]).most_common(1)[0][0]
        for i in range(len(y_test))]
    pred_all_test_acc_mv = sum(np.equal(test_y_array, pred_all_test_mv)) / len(y_test)
    print("2 class, mv for " + category + " all model testing acc: " + str(
        pred_all_test_acc_mv))
    print('#' * 50)
    print('#' * 50)












