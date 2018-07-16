# -*- coding: utf-8 -*-

import pickle
from train import line_to_features

def add_ne_tag(input_path, model):
	ans = ''
	features = []
	f = open(input_path, 'r')
	for line in f:
		feature = line_to_features(line)
		features.append(feature)

	predicted = model.predict(features)

	l = 0
	ne_list = []

	for line in features:
		p = 0
		for word in line:
			if predicted[l][p] == 'ne':
				ne_list.append('<NE>' + word['cur'] + '</NE>')
			else:
				ne_list.append(word['cur'])
			p += 1
		l += 1
	ans = '|'.join(ne_list)

	return ans