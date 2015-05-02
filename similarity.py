# -*- coding: utf-8 -*-
"""
WSD by maximizing similarity. 
"""

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic

from utils import lemmatize_sentence, lemmatize
import sys
import nltk
from nltk import word_tokenize, pos_tag

SV_SENSE_MAP = {
    "HARD1": ["difficult.a.01",
              "unmanageable.a.02"],    # not easy, requiring great physical or mental
    "HARD2": ["hard.a.02",          # dispassionate
              "difficult.a.01",
              "unvoiced.a.01",
              "arduous.s.01","intemperate.s.03","hard.s.10","hard.s.11","hard.s.12","hard.r.04",
              "hard.r.05","hard.r.07","hard.r.02"],
    "HARD3": ["hard.a.03",
              "hard.a.08",
              "hard.r.10",              
              "hard.r.01",
              "hard.r.08"],         # resisting weight or pressure
    "interest_1": ['interest.n.01'], # readiness to give attention
    "interest_2": ['interest.n.03'], # quality of causing attention to be given to
    "interest_3": ["pastime.n.01"],  # activity, etc. that one gives attention to
    "interest_4": ["sake.n.01",
                   "sake.n.03"],     # advantage, advancement or favor
    "interest_5": ["interest.n.05"], # a share in a company or business
    "interest_6": ["interest.n.06",
                   "interest.n.04"], # money paid for the use of money
    "cord": ["line.n.18","channel.n.05","line.n.07","line.n.04","line.n.06"],          # something (as a cord or rope) that is long and thin and flexible
    "formation": ["line.n.01","line.n.23"], # a formation of people or things one beside another
    "text": ["line.n.05","line.n.27","cable.n.02","note.n.02","line.n.03"],                 # text consisting of a row of words written across a page or computer screen
    "phone": ["telephone_line.n.02","telephone_wire.n.01"],   # a telephone connection
    "product": ["line.n.22","line.n.16","agate_line.n.01","line.v.02","line.v.05","occupation.n.01",
                "occupation.n.01","cable.n.02","tune.n.01","line.n.20","line.n.23","line.n.07","credit_line.n.01",
                "wrinkle.n.01","trace.v.02","lineage.n.01","production_line.n.01",
                "line.n.11"],       # a particular kind of product or merchandise
    "division": ["line.n.29","line.v.01","course.n.02"],      # a conceptual separation or distinction
    "SERVE12": ["serve.v.02","serve.v.13","serve.v.10","serve.v.15",
                "serve.v.03"],       # do duty or hold offices; serve in a specific function
    "SERVE10": ["serve.n.01","serve.v.06","serve.v.11","serve.v.05","suffice.v.01"], # provide (usually but not necessarily food)
    "SERVE2": ["serve.v.01","serve.v.07","serve.v.09"],       # serve a purpose, role, or function
    "SERVE6": ["service.v.01","service.v.02","servicing.n.01","service.n.01","service.n.09","service.n.04","service.n.05","serve.v.14"]      # be used by; as of a utility
}

def similarity_by_path(sense1, sense2, option="path"):
    """ Returns maximum path similarity between two senses. """
    if option.lower() in ["path", "path_similarity"]: # Path similaritys
        return max(wn.path_similarity(sense1,sense2), 
                   wn.path_similarity(sense1,sense2))
    elif option.lower() in ['lch', "leacock-chordorow"]: # Leacock-Chodorow
        if sense1.pos != sense2.pos: # lch can't do diff POS
            return 0
        return wn.lch_similarity(sense1, sense2)

def similarity_by_infocontent(sense1, sense2, option):
    """ Returns similarity scores by information content. """
    if sense1.pos != sense2.pos: # infocontent sim can't do diff POS.
        return 0

    if option in ['res', 'resnik']:
        return wn.res_similarity(sense1, sense2, wnic.ic('ic-bnc-resnik-add1.dat'))
    
    elif option in ['jcn', "jiang-conrath"]:
        return wn.jcn_similarity(sense1, sense2, wnic.ic('ic-bnc-add1.dat'))
  
    elif option in ['lin']:
        return wn.lin_similarity(sense1, sense2, wnic.ic('ic-bnc-add1.dat'))

def sim(sense1, sense2, option="path"):
    """ Calculates similarity based on user's choice. """
    option = option.lower()
    if option.lower() in ["path", "path_similarity", 
                        'lch', "leacock-chordorow"]:
        return similarity_by_path(sense1, sense2, option) 
    elif option.lower() in ["res", "resnik",
                          "jcn","jiang-conrath",
                          "lin"]:
        return similarity_by_infocontent(sense1, sense2, option)

def max_similarity(context_sentence, ambiguous_word,pos=None,option="path"):
    """
    Perform WSD by maximizing the sum of maximum similarity between possible 
    synsets of all words in the context sentence and the possible synsets of the 
    ambiguous words"""
    #ambiguous_word = lemmatize(ambiguous_word)
    context_sentence = lemmatize_sentence(context_sentence)
    result = {}
    for i in wn.synsets(ambiguous_word):
        #print i
        try:
            if pos and pos != str(i.pos()):
                continue
        except:
            if pos and pos != str(i.pos):
                continue 
        result[i] = sum(max([sim(i,k,option) for k in wn.synsets(j)]+[0]) \
                        for j in context_sentence)
    
    if option in ["res","resnik"]: # lower score = more similar
        result = sorted([(v,k) for k,v in result.items()])
    else: # higher score = more similar
        result = sorted([(v,k) for k,v in result.items()],reverse=True)
    #print result
    if result:
     return result[0]


    
def startSimilarity(path,confusion_matrix=True):
    actual_sense = []
    pred_sense = []
    #print sense
    #return
    from lesk_wsd import adapted_lesk

    #print path
    with open(path) as f:
        count =0
        correct_yes =0
        total =0
        for sent in f:
            #sent = sent.decode('utf8')
            count +=1
            gold,sent = sent.split("\t")
            actual_sense.append(gold)
            
            ambiguous_word = []
            if gold in SV_SENSE_MAP:
                ambiguous_word = SV_SENSE_MAP[gold][0]
            else:
                continue
            #Paper2
            #option1: jcn,#option2: lin,#option3: res,#option4: lch
            ambiguous_word = ambiguous_word.split(".")[0]
            #print ambiguous_word
            
            #Paper2
            feature2 = max_similarity
            
            #Paper1
            feature1 = adapted_lesk
            result = feature2(sent,ambiguous_word, pos='n',option="res")
            

            if result and len(result)>1:
                result = str(result[1]).replace("Synset","")
                result = result.strip("("+")"+"'")
                #print result
                sense =[k for k, v in SV_SENSE_MAP.iteritems() if result in v]
                if sense:
                    #print sense[0]
                    pred_sense.append(sense[0])
                    if sense[0]==gold:
                        correct_yes+=1
                    total+=1                    
                else:
                    print "sense map error"
                    #print result
                    #print gold
                    pred_sense.append(gold)
            else:
                print "Algo not able to judge sense"
                #pred_sense.append(wn.synsets(ambiguous_word, 'n')[0])
            if count ==100:
                break
    actual_sense = actual_sense[:len(pred_sense)]
    #print actual_sense
    #print pred_sense
    accuracy = float(correct_yes)/total 
    print "Accuracy: " ,accuracy

    if  confusion_matrix==True:
        cm = nltk.ConfusionMatrix(actual_sense,pred_sense)
        print cm
        return cm

#startSimilarity("test_data_hard.pos.tsv")
