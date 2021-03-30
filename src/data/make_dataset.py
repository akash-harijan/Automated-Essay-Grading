# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os


class DataLoader:
    def __init__(self, data_path, emb_path):
        self.data_path = data_path
        self.emb_path = emb_path

    self.word2vec_dict = {}
    with open(self.emb_path) as f:
        for line in f:
            l = line.split()
            word = l[0]
            word_emb = np.array(l[1:], dtype=np.float64)
            self.word2vec_dict[word] = word_emb

    def text_2_num(self, essay_id):

        train_x = []
        for i in range(trainx.shape[0]):
            sentence = trainx[i]
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
        return train_x


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
