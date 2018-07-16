# -*- coding: utf-8 -*-
from ne_detect import *

if __name__ == '__main__':
    model_file_name = 'model_1531730686.pickle'
    with open('./models/{}'.format(model_file_name), 'rb') as handle:
        MODEL = pickle.load(handle)

    file_path = 'text_example.txt'
    text_with_ne = add_ne_tag(file_path, MODEL)
    print(text_with_ne)