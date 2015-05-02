#!/usr/bin/env python -*- coding: utf-8 -*-
#
# Python Word Sense Disambiguation (pyWSD): Misc utility functions
#
# Copyright (C) 2014-2015 alvations
# URL:
# For license information, see LICENSE.md

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize

def remove_tags(text):
  """ Removes <tags> in angled brackets from text. """
  import re
  tags = {i:" " for i in re.findall("(<[^>\n]*>)",text.strip())}
  no_tag_text = reduce(lambda x, kv:x.replace(*kv), tags.iteritems(), text)
  return " ".join(no_tag_text.split())
  

porter = PorterStemmer()
wnl = WordNetLemmatizer()

def lemmatize(ambiguous_word, pos=False,
              lemmatizer=wnl, stemmer=porter):
    """
    Tries to convert a surface word into lemma, and if lemmatize word is not in
    wordnet then try and convert surface word into its stem.
    
    This is to handle the case where users input a surface word as an ambiguous 
    word and the surface word is a not a lemma.
    """
    lemma = lemmatizer.lemmatize(ambiguous_word)
    if pos:
        lemma = lemmatizer.lemmatize(ambiguous_word, pos=pos)
    else:
        lemma = lemmatizer.lemmatize(ambiguous_word)
    if not wn.synsets(lemma):
        #print "Lemma not found"
        #print lemma
        return ambiguous_word
    else:
     return lemma
 

def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,
                  'VB':wn.VERB, 'RB':wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''

def lemmatize_sentence(sentence,
                       tokenizer=word_tokenize, postagger=pos_tag, 
                       lemmatizer=wnl, stemmer=porter):
    lemmas = []
    for word, pos in postagger(tokenizer(sentence)):
        pos = penn2morphy(pos)
        #print pos
        lemmas.append(lemmatize(word.lower(), pos,
                                lemmatizer, stemmer))       
    return lemmas