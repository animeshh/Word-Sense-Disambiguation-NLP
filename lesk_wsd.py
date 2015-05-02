#!/usr/bin/env python -*- coding: utf-8 -*-
#
# Python Word Sense Disambiguation (pyWSD)
#
# Copyright (C) 2014-2015 alvations
# URL:
# For license information, see LICENSE.md

import string
from itertools import chain

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag

from utils import lemmatize_sentence
import sys

def get_pos_of_ambiguous_word(context_sentence, ambiguous_word):
    return {tok.lower():pos for tok, pos in 
            pos_tag(word_tokenize(context_sentence))}[ambiguous_word][0].lower()


def simple_signature(ambiguous_word,lemma=True,\
                     hyperhypo=True, stop=True):
    """ 
    Returns a synsets_signatures dictionary that includes signature words of a 
    sense from its:
    (i)   definition
    (ii)  example sentences
    (iii) hypernyms and hyponyms
    """
    synsets_signatures = {}
    for ss in wn.synsets(ambiguous_word):
        signature = []
        # Includes definition.
        try: signature+= ss.definition().split()
        except: signature+= ss.definition.split()
        # Includes examples
        try: signature+= list(chain(*[i.split() for i in ss.examples()]))
        except: signature+= list(chain(*[i.split() for i in ss.examples]))
        # Includes lemma_names.
        try: signature+= ss.lemma_names()
        except: signature+= ss.lemma_names
        # Optional: includes lemma_names of hypernyms and hyponyms.
        if hyperhypo == True:
            try: signature+= list(chain(*[i.lemma_names() for i \
                                          in ss.hypernyms()+ss.hyponyms()]))
            except: signature+= list(chain(*[i.lemma_names for i \
                                             in ss.hypernyms()+ss.hyponyms()]))
        # Optional: removes stopwords.
        if stop == True:
            signature = [i for i in signature if i not in stopwords.words('english')]
        synsets_signatures[ss] = signature
        
    return synsets_signatures




def compare_overlaps(context, synsets_signatures):
    """ 
    Calculates overlaps between the context sentence and the synset_signture
    and returns a ranked list of synsets from highest overlap to lowest.
    """
    overlaplen_synsets = [] # a tuple of (len(overlap), synset).
    for ss in synsets_signatures:
        overlaps = set(synsets_signatures[ss]).intersection(context)
        overlaplen_synsets.append((len(overlaps), ss))
    
    # Rank synsets from highest to lowest overlap.
    ranked_synsets = sorted(overlaplen_synsets, reverse=True)
    
    # Normalize scores such that it's between 0 to 1. 
    #total = float(sum(i[0] for i in ranked_synsets))
    #ranked_synsets = [(i/total,j) for i,j in ranked_synsets]
      
    # Returns only the best sense.
    return ranked_synsets[0]

def adapted_lesk(context_sentence, ambiguous_word, \
                pos=None, option=False,lemma=True,hyperhypo=True, \
                stop=True):
    """
    This function is the implementation of the Adapted Lesk algorithm, 
    described in Banerjee and Pederson (2002). It makes use of the lexical 
    items from semantically related senses within the wordnet 
    hierarchies and to generate more lexical items for each sense. 
    see www.d.umn.edu/~tpederse/Pubs/cicling2002-b.pdf‎
    """
    # Ensure that ambiguous word is a lemma.
    #ambiguous_word = lemmatize(ambiguous_word)
    # Get the signatures for each synset.

    ss_sign = simple_signature(ambiguous_word,lemma=True,hyperhypo=True)
    #print ss_sign
    for ss in ss_sign:
        related_senses = list(set(ss.member_holonyms() + ss.member_meronyms() + 
                                 ss.part_meronyms() + ss.part_holonyms() + 
                                 ss.similar_tos() + ss.substance_holonyms() + 
                                 ss.substance_meronyms()))
    
        try:
            signature = list([j for j in chain(*[i.lemma_names() for i in \
                      related_senses]) if j not in stopwords.words('english')])
        except:
            signature = list([j for j in chain(*[i.lemma_names for i in \
                      related_senses]) if j not in stopwords.words('english')])
    ss_sign[ss]+=signature
  
    context_sentence = lemmatize_sentence(context_sentence)
    best_sense = compare_overlaps(context_sentence, ss_sign)
    return best_sense