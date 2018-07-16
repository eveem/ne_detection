# -*- coding: utf-8 -*-

import pickle
from train import line_to_features

features = []
f = open('./text_test.txt', 'r')
for line in f:
	feature = line_to_features(line)
	features.append(feature)

model_file_name = 'model_1531730686.pickle'
with open('./models/{}'.format(model_file_name), 'rb') as handle:
    MODEL = pickle.load(handle)

predicted = MODEL.predict(features)

l = 0
ne_list = []

for line in features:
	p = 0
	for word in line:
		if predicted[l][p] == 'ne':
			ne_list.append(word['cur'])
		p += 1
	l += 1

print(ne_list)
