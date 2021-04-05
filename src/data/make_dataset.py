# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os


class DataLoader:
    def __init__(self, data_path, emb_path):
        self.data_path = data_path
        self.emb_path = emb_path

        self.whole_data = pd.read_csv(self.data_path, sep='\t', encoding='unicode_escape')
        self.whole_data = self.whole_data.fillna(0)
        self.total_essay_sets = self.whole_data.essay_set.unique()

        print("DataSet is loaded with {0} Shape".format(self.whole_data.shape))

        self.word2vec_dict = {}
        with open(self.emb_path) as f:
            for line in f:
                l = line.split()
                word = l[0]
                word_emb = np.array(l[1:], dtype=np.float64)
                self.word2vec_dict[word] = word_emb

    def get_essay_data(self, essay_id):
        essay_ds = self.whole_data[self.whole_data.essay_set == essay_id]
        essay_ds = np.array(essay_ds)
        print("Dataset with Essay id {0} has {1} Shape".format(essay_id, essay_ds.shape))

        essay_text = np.array(essay_ds[:, 2])
        essay_score = np.array(essay_ds[:, 5])

        train_y = essay_score.reshape(essay_score.shape[0], 1)

        print("Test X ", essay_text[0])
        print("Test Y ", essay_score[0])

        train_x = []
        for i in range(essay_text.shape[0]):
            sentence = essay_text[i]
            emb = np.zeros(300)
            words = sentence.split()
            t_n = 0
            skip = 0
            for word in words:
                try:
                    emb += word2vec_dict[word.lower()]
                    t_n += 1
                except:
                    skip += 1
                    pass
            train_x.append(emb / t_n)

        return np.array(train_x), np.array(train_y)

    def get_data(self):
        train_X, train_Y, val_X, val_Y, test_X, test_Y = [], [], [], [], [], []
        for i in self.total_essay_sets:
            x, y = self.get_essay_data(i)
            split_1 = int(x.shape[0]*0.8)
            split_2 = int(x.shape[0]*0.9)
            train_X.append(x[:split_1])
            train_Y.append(y[:split_1])
            val_X.append(x[split_1:split_2])
            val_Y.append(y[split_1:split_2])
            test_X.append(x[split_2:])
            test_Y.append(y[split_2:])

        train_X = np.vstack(train_X)
        train_Y = np.vstack(train_Y)

        val_X = np.vstack(val_X)
        val_Y = np.vstack(val_Y)

        test_X = np.vstack(test_X)
        test_Y = np.vstack(test_Y)

        print("Train X : {0}, Train Y : {1}, val X : {2}, val Y : {3}, test X : {4}, test Y : {5}".
              format(train_X.shape, train_Y.shape, val_X.shape, val_Y.shape, test_X.shape, test_Y.shape))


if __name__ == '__main__':

    dataset_path = "/home/akash/Pet/Auto Essay Grading-old/Auto Essay Grading/Akash/Data/training_set_rel3.tsv"
    emb_path = "/home/akash/Pet/Auto Essay Grading-old/Auto Essay Grading/automated-essay-grading/glove/glove.42B.300d.txt"
    dataset = DataLoader(dataset_path, emb_path)

    dataset.get_data()
