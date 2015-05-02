# -*- coding: utf-8 -*-
from __future__ import division
import nltk
import random
import sys
from nltk.corpus import senseval
from nltk.classify import accuracy, NaiveBayesClassifier, MaxentClassifier
from collections import defaultdict
from similarity import startSimilarity

def senses(word):
    """
    This takes a target word from senseval-2 (find out what the possible
    are by running senseval.fileides()), and it returns the list of possible 
    senses for the word
    """
    return list(set(i.senses[0] for i in senseval.instances(word)))


    

def sense_instances(instances, sense):
    """
    This returns the list of instances in instances that have the sense sense
    """
    return [instance for instance in instances if instance.senses[0]==sense]



_inst_cache = {}

STOPWORDS = ['.', ',', '?', '"', '``', "''", "'", '--', '-', ':', ';', '(',
             ')', '$', '000', '1', '2', '10,' 'I', 'i', 'a', 'about', 'after', 'all', 'also', 'an', 'any',
             'are', 'as', 'at', 'and', 'be', 'being', 'because', 'been', 'but', 'by',
             'can', "'d", 'did', 'do', "don'", 'don', 'for', 'from', 'had','has', 'have', 'he',
             'her','him', 'his', 'how', 'if', 'is', 'in', 'it', 'its', "'ll", "'m", 'me',
             'more', 'my', 'n', 'no', 'not', 'of', 'on', 'one', 'or', "'re", "'s", "s",
             'said', 'say', 'says', 'she', 'so', 'some', 'such', "'t", 'than', 'that', 'the',
             'them', 'they', 'their', 'there', 'this', 'to', 'up', 'us', "'ve", 'was', 'we', 'were',
             'what', 'when', 'where', 'which', 'who', 'will', 'with', 'years', 'you',
             'your']

NO_STOPWORDS = []

def wsd_context_features(instance, vocab, dist=3):
    #print "here...."
    features = {}
    ind = instance.position
    con = instance.context
    for i in range(max(0, ind-dist), ind):
        j = ind-i
        features['left-context-word-%s(%s)' % (j, con[i][0])] = True

    for i in range(ind+1, min(ind+dist+1, len(con))):
        j = i-ind
        features['right-context-word-%s(%s)' % (j, con[i][0])] = True

 
    features['word'] = instance.word
    features['pos'] = con[1][1]
    return features



def wsd_word_features(instance, vocab, dist=3):
    """
    Create a featureset where every key returns False unless it occurs in the
    instance's context
    """
    #print "chugggggg"
    features = defaultdict(lambda:False)
    features['alwayson'] = True
    #cur_words = [w for (w, pos) in i.context]
    try:
        for(w, pos) in instance.context:
            if w in vocab:
                features[w] = True
    except ValueError:
        pass
    return features


def extract_vocab(instances, stopwords=STOPWORDS, n=600):
    """
    Given a list of senseval instances, return a list of the n most frequent words that
    appear in the context of instances.  The context is the sentence that the target word
    appeared in within the corpus.
    """
    #cfd = nltk.ConditionalFreqDist()
    fd = nltk.FreqDist()
    for i in instances:
        (target, suffix) = i.word.split('-')
        try:
            words = [w for (w, pos) in i.context if not w == target]
        except ValueError:
            pass
        for word in set(words) - set(stopwords):
            fd[word]+=1 
            #for sense in i.senses:
                #cfd[sense].inc(word)
    return sorted(fd.keys()[:n+1])

def extract_vocab_frequency(instances, stopwords=STOPWORDS, n=300):
    """
    Given a list of senseval instances, return a list of the n most frequent words that
    appears in its context (i.e., the sentence with the target word in), output is in order
    of frequency and includes also the number of instances in which that key appears in the
    context of instances.
    """
    #cfd = nltk.ConditionalFreqDist()
    fd = nltk.FreqDist()
    for i in instances:
        (target, suffix) = i.word.split('-')
        try:
            words = [w for (w, pos) in i.context if not w == target]
        except ValueError:
            pass
        for word in set(words) - set(stopwords):
            fd.inc(word) 
            #for sense in i.senses:
                #cfd[sense].inc(word)
    return fd.items()[:n+1]
        
  
    
def wst_classifier(trainer, word, features, stopwords_list = STOPWORDS, number=600, log=False, distance=3, confusion_matrix=False):
    """
    This function takes as arguments:
        a trainer (e.g., NaiveBayesClassifier.train);
        a target word from senseval2 (you can find these out with senseval.fileids(),
            and they are 'hard.pos', 'interest.pos', 'line.pos' and 'serve.pos');
        a feature set (this can be wsd_context_features or wsd_word_features);
        a number (defaults to 300), which determines for wsd_word_features the number of
            most frequent words within the context of a given sense that you use to classify examples;
        a distance (defaults to 3) which determines the size of the window for wsd_context_features (if distance=3, then
            wsd_context_features gives 3 words and tags to the left and 3 words and tags to
            the right of the target word);
        log (defaults to false), which if set to True outputs the errors into a file errors.txt
        confusion_matrix (defaults to False), which if set to True prints a confusion matrix."""

    print "Reading data..."
    global _inst_cache
    #print "",senseval.instances(word)[0]
    if word not in _inst_cache:
        _inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
    events = _inst_cache[word][:]
    senses = list(set(l for (i, l) in events))
    instances = [i for (i, l) in events]

    vocab = extract_vocab(instances, stopwords=stopwords_list, n=number)
    print ' Senses: ' + ' '.join(senses)

    # Split the instances into a training and test set,
    #if n > len(events): n = len(events)
    n = len(events)
    print n
    #random.seed(5444522)
    random.shuffle(events)
    training_data = events[:int(0.9 * n)]
    test_data = events[int(0.1 * n):n]
    path = "test_data_"+ word +".tsv"
    
    #creating test data for similarity algo
    with open(path,"w") as f:
        for (text,label) in test_data:
            sentences = ""
            for wordTag in text.context:
                if len(wordTag)==2:
                    word,tag = wordTag
                    sentences +=word +" "
            f.write(""+label+"\t"+sentences +"\n")
    
    startSimilarity(path)

    # Train classifier
    print 'Training classifier...'
    #print training_data[0]
    classifier = trainer([(features(i, vocab, distance), label) for (i, label) in training_data])
    # Test classifier
    print 'Testing classifier...'
    acc = accuracy(classifier, [(features(i, vocab, distance), label) for (i, label) in test_data] )
    print 'Accuracy: %6.4f' % acc
    
    if confusion_matrix==True:
        gold = [label for (i, label) in test_data]
        derived = [classifier.classify(features(i,vocab)) for (i,label) in test_data]
        #print derived
        cm = nltk.ConfusionMatrix(gold,derived)
        print "Machine Learning Confusion-Matrix" 
        print cm
        return cm
        
        
    
def start():
    
    #wsd_word_features =  wsd_word_features(instance, vocab, dist=3)
    #wst_classifier(NaiveBayesClassifier.train, 'interest.pos', wsd_word_features,distance=3, confusion_matrix=True)
    #wst_classifier(NaiveBayesClassifier.train, 'hard.pos', wsd_word_features,distance=3, confusion_matrix=True)
    #wst_classifier(NaiveBayesClassifier.train, 'line.pos', wsd_word_features,distance=3, confusion_matrix=True)
    #wst_classifier(NaiveBayesClassifier.train, 'serve.pos', wsd_word_features,distance=3, confusion_matrix=True)

    """wst_classifier(NaiveBayesClassifier.train, 'interest.pos', wsd_context_features,distance=3, confusion_matrix=True)
    wst_classifier(NaiveBayesClassifier.train, 'hard.pos', wsd_context_features,distance=3, confusion_matrix=True)
    wst_classifier(NaiveBayesClassifier.train, 'line.pos', wsd_context_features,distance=3, confusion_matrix=True)
    wst_classifier(NaiveBayesClassifier.train, 'serve.pos', wsd_context_features,distance=3, confusion_matrix=True)"""
    

    # logistic regression ===  max Entropy classifier
    wst_classifier(MaxentClassifier.train, 'interest.pos', wsd_word_features,distance=3, confusion_matrix=True)
    """wst_classifier(MaxentClassifier.train, 'hard.pos', wsd_word_features,distance=3, confusion_matrix=True)
    wst_classifier(MaxentClassifier.train, 'line.pos', wsd_word_features,distance=3, confusion_matrix=True)
    wst_classifier(MaxentClassifier.train, 'serve.pos', wsd_word_features,distance=3, confusion_matrix=True)"""
   
    
    """wst_classifier(MaxentClassifier.train, 'interest.pos', wsd_context_features,distance=3, confusion_matrix=True)
    wst_classifier(MaxentClassifier.train, 'hard.pos', wsd_context_features,distance=3, confusion_matrix=True)
    wst_classifier(MaxentClassifier.train, 'line.pos', wsd_context_features,distance=3, confusion_matrix=True)
    wst_classifier(MaxentClassifier.train, 'serve.pos', wsd_context_features,distance=3, confusion_matrix=True)"""
    

start()

# Frequency Baseline
sense_fd = nltk.FreqDist([i.senses[0] for i in senseval.instances('hard.pos')])
most_frequent_sense = sense_fd.keys()[0]
frequency_sense_baseline = sense_fd.freq(sense_fd.keys()[0])
print "frequency baseline:" ,frequency_sense_baseline
##0.79736902838679902
