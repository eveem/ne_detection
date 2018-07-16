import time
import pickle
import sklearn_crfsuite

def line_to_features(line):
    # (current, p1_word, p2_word, p3_word, p4_word, n1_word, n2_word, n3_word, n4_word, isnumcur, label)

    features = []
    words = line.split('|')
    len_words = len(words)

    for idx, word in enumerate(words):
        lb = 'o'
        temp = word
        if '<NE>' in word:
            temp = word.replace('<NE>', '')
            temp = word.replace('/<NE>', '')
            lb = 'ne'
        
        feature = {
            'cur': temp,
            'isnumcur': temp.isdigit(),
            'label': lb
        }

        for i in range(1, 4):
            if idx - i >= 0:
                feature['p{0}_word'.format(i)] = words[idx - i]
            else:
                feature['p{0}_word'.format(i)] = '**'
            if idx + i < len_words:
                feature['n{0}_word'.format(i)] = words[idx + i]
            else:
                feature['n{0}_word'.format(i)] = '**'
        features.append(feature)
    return features
     
def raw_train_to_features(file_path):
    list_of_features_by_line_by_word = []
    f = open(file_path, 'r')

    for idx, line in enumerate(f):
        if idx % 1000 == 0:
            print(idx)
        feature = line_to_features(line)
        list_of_features_by_line_by_word.append(feature)
    return list_of_features_by_line_by_word

def train(features):
    y_trains = []
    x_trains = []
    for list_of_feature_by_line in features:
        y_train = []
        x_train = []
        for feature_by_char in list_of_feature_by_line:
            y_train.append(feature_by_char['label'])
            feature_by_char.pop('label')
            x_train.append(feature_by_char)
        y_trains.append(y_train)
        x_trains.append(x_train)
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1 = 0.1,
        c2 = 0.1,
        max_iterations = 100,
        all_possible_transitions = True
    )
    crf.fit(x_trains, y_trains)
    return crf
    
# features = raw_train_to_features('./train/BEST/all_best.txt')
# MODEL = train(features)
# model_file_name = 'model_{}.pickle'.format(int(time.time()))
# with open('./models/{}'.format(model_file_name), 'wb') as handle:
#     pickle.dump(MODEL, handle, protocol = pickle.HIGHEST_PROTOCOL)