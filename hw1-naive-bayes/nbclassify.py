##basic imports
import os
import string
import time
import sys
import json
import numpy as np
import math
from string import digits

#################################################ALL STOPWORD#############################################################
stopwords = {'should', 'has', 'yourselves', 'through', 'am', 'don', 'is', 'it', 'this', 'before', 'during', 'whom', 'their', 'yours', 'above', 't', 'them', 'again', 'then', 'been', 'all', 'him', 'own', 'both', 'doing', 'or', 'about', 'few', 'because', 'our', 'which', 'me', 'same', 'she', 'he', 'when', 'how', 'that', 'between', 'can', 'down', 'its', 'not', 'be', 'have', 's', 'very', 'did', 'ours', 'was', 'on', 'under', 'below', 'those', 'too', 'but', 'his', 'who', 'why', 'having', 'herself', 'in', 'they', 'other', 'does', 'myself', 'by', 'as', 'here', 'ourselves', 'out', 'off', 'will', 'hers', 'what', 'such', 'her', 'for', 'more', 'after', 'to', 'while', 'of', 'theirs', 'into', 'further', 'from', 'than', 'themselves', 'if', 'yourself', 'no', 'do', 'over', 'were', 'are', 'and', 'at', 'any', 'where', 'being', 'these', 'you', 'himself', 'most', 'there', 'a', 'itself', 'just', 'up', 'my', 'once', 'only', 'nor', 'against', 'until', 'your', 'an', 'with', 'we', 'had', 'so', 'now', 'each', 'the', 'i', 'some'}

######################################ALL VARIABLES############################################################################
transformation1_chrs = {32: '',33: '', 34: '', 35: '', 36: '', 37: '', 38: '', 39: '', 40: '', 41: '', 42: '', 43: '', 44: '', 45: '', 46: '', 47: '', 58: '', 59: '', 60: '', 61: '', 62: '', 63: '', 64: '', 91: '', 92: '', 93: '', 94: '', 95: '', 96: '', 123: '', 124: '', 125: '', 126: ''}
#transformation2_digits = {48: '', 49: '', 50: '', 51: '', 52: '', 53: '', 54: '', 55: '', 56: ''}
transformation2_digits = str.maketrans('', '', digits)
test_path = sys.argv[1]
test_path_length = len(test_path.split("/"))
output_file = 'nboutput.txt'
model_file = "nbmodel.txt"

prior_probs = {}
cond_probs = {}
predictions = []
total_eg = {'egs':0}
correct = {'egs':0}
#######################################FUNCTION TO READ MODEL FILE#################################################
def read_model_file():
    with open(model_file) as json_file:
        dicts = json.load(json_file)

    prior_probs = dicts['prior_prob']
    cond_probs = dicts['cond_prob']

    return prior_probs,cond_probs
    #print(prior_probs,cond_probs)
##############################################PREDICTION FUNCTION###################################################
def predict(lines,file_path):
    #print(prior_probs)
    bayes_score = {'truthful':math.log(prior_probs['truthful'],10), 'deceptive':math.log(prior_probs['deceptive'],10), 'positive':math.log(prior_probs['positive'],10), 'negative':math.log(prior_probs['negative'],10)}

    #print(line)
    line = lines.split(" ")
    total_eg['egs']+=1
    for word in line:
        wordx = word.lower().strip()

        #rdword = wordx.translate(transformation2_digits)
        rdl = wordx.split(",")

        for wrd in rdl:
            new_word = wrd.translate(transformation1_chrs)

            if new_word == "":
                continue

            elif new_word in stopwords:
                continue

            for c in bayes_score.keys():
                if new_word in cond_probs[c]:
                    curr_prob = cond_probs[c][new_word]
                    prev_prob = bayes_score[c]

                    bayes_score[c] = prev_prob + math.log(curr_prob,10)


    thispred = []
    if bayes_score['truthful'] > bayes_score['deceptive']:
        thispred.append('truthful')

    else:
        thispred.append('deceptive')


    if bayes_score['positive'] > bayes_score['negative']:
        thispred.append('positive')

    else:
        thispred.append('negative')

    thispred.append(file_path)

    predictions.append(thispred)

    #temp = file_path.split("\\")
    #var2 = temp[1].split('_')[0]
    #var1 = temp[2].split('_')[0]
    #if var1 == thispred[0] and var2 == thispred[1]:
        #correct['egs']+=1

##########################################FUNCTION TO READ ALL FILES##############################
def read_content():
    for root,directories,files in os.walk(test_path,topdown= False):
        for name in directories:
            dir_split = os.path.join(root,name).split("/")

            if len(dir_split) == test_path_length+3:
            #if len(name) > 4 and name[len(name)-4:] == 'fold':
            ##if len(dir_split)==15
                #print(dir_split)
                doc_files = [f for f in os.listdir(os.path.join(root,name))]

                for doc_name in doc_files:
                    doc_file = os.path.join(root,name)+"/"+doc_name
                    f = open(doc_file,"r")
                    #print(f.read())
                    predict(f.read(),doc_file)

##############################FUNCTION TO MAKE OUTPUT FILE#######################################
def make_output_file():
    f = open("nboutput.txt","w")
    #print(predictions)
    for x in predictions:
        f.write(' '.join(x)+'\n')

    f.close()
#########################CALLING FUNCTIONS##################################################
prior_probs,cond_probs = read_model_file()

read_content()
make_output_file()
#print(correct['egs']/total_eg['egs'])
#########################################################################################
