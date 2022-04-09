# -*- coding: utf-8 -*-
"""
Semantic Relations in Knowledge Repositories and in Texts

Final Project

Julia Evans
"""
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.svm import SVC
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import en_core_web_lg
import spacy


parse = spacy.load('en_core_web_sm')           # Dependensy parser 
nlp = en_core_web_lg.load()                    # Word embeddings
stop_words = set(stopwords.words('english'))   # Stopwords


# Make dictionary of WordNet sense codes and synset offsets
synset_offsets = { }
with open('index.sense') as f:
    for line in f.readlines():
        line = line.split()
        synset_offsets['"' + line[0] + '"'] = line[1]
        

# Make dictionary of perceptual features
perceptual_features = { }
with open('norms.dat') as f:
    for line in f.readlines():
        line = line.split('\t')
        concept = line[2]
        feature = line[3]
        
        if concept in perceptual_features.keys():
            perceptual_features[concept].append(feature)
        else:
            perceptual_features[concept] = [feature]


def get_hypernyms(entity_offset):
    """
    Find WordNet hypernyms for entities
    """
    if entity_offset in synset_offsets.keys():
        offset = synset_offsets[entity_offset]
        try:
            synset = wn.synset_from_pos_and_offset('n', int(offset))
            hyper_sets = synset.hypernyms()
            hypernyms = [ ]
            for h in hyper_sets:
                hypernyms += [h.name()]
        except:
            hypernyms = [0]
    else:
        hypernyms = [0]
        
    return hypernyms


def get_synonym_feats(entity_offset):
    """
    Find WordNet synonyms for entity, perceptual features of synonyms
    """
    if entity_offset in synset_offsets.keys():
        offset = synset_offsets[entity_offset]
        try:
            synset = wn.synset_from_pos_and_offset('n', int(offset))
            for l in synset.lemmas():
                    if l.name() in perceptual_features.keys():
                        synonym_feats = perceptual_features[l.name()]
                    else:
                        synonym_feats = [0]
        except:
            synonym_feats = [0]
    else:
        synonym_feats = [0]

    return synonym_feats


def encode_features(lst):
    """
    Encode features as real numbers
    """
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(lst)
    encoded_lst = label_encoder.transform(lst)

    return encoded_lst


def encode_list_of_lists(lst_of_lsts, length):
    """
    Encode features represented as list of lists
    """
    
    all_features = [ ]
    
    for lst in lst_of_lsts:
        for feature in lst:
            all_features.append(feature)
    
    all_features = list(set(all_features))
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(all_features)
    encoding = label_encoder.transform(all_features)
    
    encoding_dict = { }
    
    for i in range(len(all_features)):
        encoding_dict[all_features[i]] = encoding[i]
        
    encoded_features = [ ]
    longest = 0
    
    for lst in lst_of_lsts:
        feature_lst = [ ]
        for feature in lst:
            feature_lst += [encoding_dict[feature]]
        lst_len = len(feature_lst)
        if lst_len > longest:
            longest = lst_len
        encoded_features.append(feature_lst)
    
    # Adjust padding to match length
    for i in range(len(encoded_features)):
        lst = encoded_features[i]
        if len(lst) < length:
            padding = length - len(lst)
            for i in range(padding):
                lst += [0]
        elif len(lst) > length:
            lst = lst[0:length]
            encoded_features[i] = lst 
    
    return encoded_features


def compile_vectors(*lsts):
    """
    Collect features from the given lists into vectors
    """
    features = [ ]
    
    for i in range(len(lsts[0])):
        feat_vec = [ ]
        for l in lsts:            
            try:  # If it's a list or an array
                feat_vec += list(l[i])
            except TypeError:  # If it's an int or a float
                feat_vec.append(l[i])        
        features.append(feat_vec)

    return features


def get_feats(filename, perceptual):
    """
    Extract the features
    """
    # Patterns
    e1_pattern = r'<e1>(.*)<\/e1>'
    e2_pattern = r'<e2>(.*)<\/e2>'
    relation_pattern = r'\(e[12], ?e[12]\) = \"(\w*)\"'
    sentence_pattern = r'[0-9][0-9][0-9] \"(.*)\"'
    replace_pattern = r'<e([12])>.*</e\1>'
    replace_tag_pattern = r'<\/?e[12]>'
    e1_wn_pattern = r'WordNet\(e1\) = (.*?)\,'
    e2_wn_pattern = r'WordNet\(e2\) = (.*?)\,'
            
    entity_one = [ ]
    entity_two = [ ]
    
    similarity_scores = [ ]
    e1_word_embeddings = [ ]
    e2_word_embeddings = [ ]

    relation_tag = [ ]
    
    parsed_sentences = [ ]
    sentence_embeddings = [ ]

    e1_hypers = [ ]
    e2_hypers = [ ]
    
    e1_percept_feats = [ ]
    e2_percept_feats = [ ]
    
    with open(filename) as f:
        data = f.read()
        data = data.split('\n\n')
        
    for sample in data:
        
        # Entities
        e1 = re.search(e1_pattern, sample)
        e2 = re.search(e2_pattern, sample)
        if e1:
            e1 = e1.group(1)
            e2 = e2.group(1)
            
            entity_one.append(e1)
            entity_two.append(e2)
            
            # Similarity scores
            if (nlp(e1).vector_norm) and (nlp(e2).vector_norm):
                similarity = nlp(e1).similarity(nlp(e2))   
                similarity_scores.append(similarity)
            else:
                similarity_scores.append(0)
            
            # Word embeddings
            e1_word_embeddings.append(nlp(e1).vector)
            e2_word_embeddings.append(nlp(e2).vector)
            

        # Labels
        relation = re.search(relation_pattern, sample)
        if relation:
            relation_tag.append(relation.group(1))        
        
        # Sentence
        sentence = re.search(sentence_pattern, sample)
        
        if sentence:
            sentence = sentence.group(1)
            
            # Get dependency parse
            full_sent = re.sub(replace_tag_pattern, '', sentence)
            doc = parse(full_sent) 
            dp = [ ] 
            for token in doc:
                parsed_tokens = (token.text, token.pos_, token.dep_)
                dp.append(str(parsed_tokens))
            parsed_sentences.append(dp)            
            
            # Extract average word embeddings
            sentence = re.sub(replace_pattern, '', sentence)
            word_tokens = word_tokenize(sentence) 
            filtered_sentence = ' '.join([w for w in word_tokens if not w in stop_words])
            sentence_embeddings.append(nlp(filtered_sentence).vector)
        
        
        # WordNet offsets
        e1_offset = re.search(e1_wn_pattern, sample)
        e2_offset = re.search(e2_wn_pattern, sample)
            
        if e1_offset:           
            e1_hyp = get_hypernyms(e1_offset.group(1))
            e1_hypers.append(e1_hyp)
            
            e2_hyp = get_hypernyms(e2_offset.group(1))
            e2_hypers.append(e2_hyp)
            
            # Perceptual features
            if e1 in perceptual_features.keys():
                e1_percept_feats.append(perceptual_features[e1])
            else:
                e1_percept = get_synonym_feats(e1_offset.group(1))
                e1_percept_feats.append(e1_percept)
                
            if e2 in perceptual_features.keys():
                e2_percept_feats.append(perceptual_features[e2])
            else:
                e2_percept = get_synonym_feats(e2_offset.group(1))
                e2_percept_feats.append(e2_percept)
                
    # Get encoded feature vectors
    entity_one = encode_features(entity_one)
    entity_two = encode_features(entity_two)
    labels = encode_features(relation_tag)
    
    # Get encoded features for lists of lists
    e1_hypers_encoded = encode_list_of_lists(e1_hypers, 80)
    e2_hypers_encoded = encode_list_of_lists(e2_hypers, 80)
    sentences_encoded = encode_list_of_lists(parsed_sentences, 100)
    e1_percept_encoded = encode_list_of_lists(e1_percept_feats, 100)
    e2_percept_encoded = encode_list_of_lists(e2_percept_feats, 100)
    
    # Get feature vectors
    if perceptual:
        feat_vectors = compile_vectors(entity_one, e1_hypers_encoded,
                                        entity_two, e2_hypers_encoded,
                                        similarity_scores, sentences_encoded,
                                        e1_word_embeddings, e2_word_embeddings,
                                        sentence_embeddings, e1_percept_encoded,
                                        e2_percept_encoded)
    else:
        feat_vectors = compile_vectors(entity_one, e1_hypers_encoded,
                                        entity_two, e2_hypers_encoded,
                                        similarity_scores, sentences_encoded,
                                        e1_word_embeddings, e2_word_embeddings,
                                        sentence_embeddings)

    return feat_vectors, labels


def get_test_train_data_split(features_lst, tags):
    """
    Split for training and testing
    """
    x = normalize_data(features_lst)
    y = tags
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

    return x_train, x_test, y_train, y_test


def normalize_data(features_lst):
    """
    Normalize data
    """
    x = np.asarray(features_lst)
    x = preprocessing.scale(x)
    
    return x


def output_labels(input_file, predictions, relation_name):
    """
    Outputs a file with the predicted label
    """
    output_file = relation_name + '-predictions.txt'
    
    with open(input_file) as f:
        data = f.read()
        data = data.split('\n\n')

    with open(output_file, 'a') as f:

        for i in range(len(data) - 1):
            
            if predictions[i] == 0:
                pred_label = '"false"'
            else:
                pred_label = '"true"'
            
            sample = data[i]
            sample = sample.replace('"?"', pred_label)
            f.write(sample + '\n\n')


class Model:

    def __init__(self, model_type):
        """
        model_type:
            svm = SVM model
            logreg = logistic regression model
        """
        
        if model_type == 'svm':
            self.model = SVC(kernel = 'rbf', gamma = 'scale', C = 10)
        elif model_type == 'logreg':
            self.model = LogisticRegression()


    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)


    def test(self, x_test):
        return self.model.predict(x_test)
    
    
    def score(self, x_test, y_test):
        return self.model.score(x_test, y_test)


if __name__ == '__main__':
    reps = 100
    train_file = 'relation-1-train.txt'
    perceptual = False
    features, labels = get_feats(train_file, perceptual)
    
    # Test the models
    lr_score = 0
    svm_score = 0
    
    for i in range(reps):
        x_train, x_test, y_train, y_test = get_test_train_data_split(features, labels)

        log_reg_model = Model(model_type = 'logreg')
        log_reg_model.train(x_train, y_train)
        lr_score += log_reg_model.score(x_test, y_test)
     
        svm_model = Model(model_type = 'svm')
        svm_model.train(x_train, y_train)
        svm_score += svm_model.score(x_test, y_test)

    print('Logistic Regression:', lr_score/reps)
    print('SVM:', svm_score/reps)
    
    
    # Get predictions
    test_file = 'relation-1-test.txt'
    x_test, dummy_y_test = get_feats(test_file, perceptual)
    
    features = normalize_data(features)
    x_test = normalize_data(x_test)
    
    log_reg_model = Model(model_type = 'logreg')
    log_reg_model.train(features, labels)

    svm_model = Model(model_type = 'svm')
    svm_model.train(features, labels)
        
    svm_predicted_labels = svm_model.test(x_test)
    lr_predicted_labels = svm_model.test(x_test)

    outfile_name = 'relation-1'
    output_labels(test_file, svm_predicted_labels, outfile_name)
