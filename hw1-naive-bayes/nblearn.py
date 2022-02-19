################################ALL IMPORTS##########################################################################
import os
import string
import time
import sys
import json
import numpy as np
import math
from string import digits
#starttime = time.time()
##############################################ALL STOPWORDS##############################################################
stopwords = {'should', 'has', 'yourselves', 'through', 'am', 'don', 'is', 'it', 'this', 'before', 'during', 'whom', 'their', 'yours', 'above', 't', 'them', 'again', 'then', 'been', 'all', 'him', 'own', 'both', 'doing', 'or', 'about', 'few', 'because', 'our', 'which', 'me', 'same', 'she', 'he', 'when', 'how', 'that', 'between', 'can', 'down', 'its', 'not', 'be', 'have', 's', 'very', 'did', 'ours', 'was', 'on', 'under', 'below', 'those', 'too', 'but', 'his', 'who', 'why', 'having', 'herself', 'in', 'they', 'other', 'does', 'myself', 'by', 'as', 'here', 'ourselves', 'out', 'off', 'will', 'hers', 'what', 'such', 'her', 'for', 'more', 'after', 'to', 'while', 'of', 'theirs', 'into', 'further', 'from', 'than', 'themselves', 'if', 'yourself', 'no', 'do', 'over', 'were', 'are', 'and', 'at', 'any', 'where', 'being', 'these', 'you', 'himself', 'most', 'there', 'a', 'itself', 'just', 'up', 'my', 'once', 'only', 'nor', 'against', 'until', 'your', 'an', 'with', 'we', 'had', 'so', 'now', 'each', 'the', 'i', 'some'}
######################################ALL VARIABLES#########################################################################
transformation1_chrs = {32:'',33: '', 34: '', 35: '', 36: '', 37: '', 38: '', 39: '', 40: '', 41: '', 42: '', 43: '', 44: '', 45: '', 46: '', 47: '', 58: '', 59: '', 60: '', 61: '', 62: '', 63: '', 64: '', 91: '', 92: '', 93: '', 94: '', 95: '', 96: '', 123: '', 124: '', 125: '', 126: ''}
#transformation2_digits = {48: '', 49: '', 50: '', 51: '', 52: '', 53: '', 54: '', 55: '', 56: ''}
transformation2_digits = str.maketrans('', '', digits)
ip_path = sys.argv[1]
ip_path_length = len(ip_path.split("/"))
output_file = 'nbmodel.txt'

priors = {'truthful':0,'deceptive':0,'positive':0,'negative':0}
prior_prob = {'truthful':0,'deceptive':0,'positive':0,'negative':0}
class_feat = {'truthful':dict(),'deceptive':dict(),'positive':dict(),'negative':dict()}
total_feat = {'total':0}
vocabulary = set()

############################################COUNTING OCCURENCE OF FEATURES IN ALL CLASSES########################
def get_count(pos_or_neg,true_or_decp,line):

    lines = line.split(" ")
    for word in lines:
        wordx = word.lower().strip()

        #rdword = wordx.translate(transformation2_digits)
        rdl = wordx.split(",")

        for wrd in rdl:
            new_word = wrd.translate(transformation1_chrs)

            if new_word == "":
                continue

            elif new_word in stopwords:
                continue

            priors[pos_or_neg]+=1
            priors[true_or_decp]+=1
            #print(new_word,pos_or_neg,true_or_decp)
            if new_word not in class_feat[pos_or_neg]:
                class_feat[pos_or_neg][new_word] = 1

            else:
                class_feat[pos_or_neg][new_word]+=1


            if new_word not in class_feat[true_or_decp]:
                class_feat[true_or_decp][new_word] = 1

            else:
                class_feat[true_or_decp][new_word]+=1


##############################################3READING ALL FILES#############################################
def read_content():

    for root,directories,files in os.walk(ip_path,topdown=False):
        for name in directories:
            #print(name)
            dir_split = os.path.join(root,name).split("/")
            #print(os.path.join(root,name),dir_split)
            
            if len(dir_split) == ip_path_length+3:
            #if len(name) > 4 and name[0:4] == 'fold':
            ##if len(dir_split)==15
                #print(dir_split)
                var1 = dir_split[-3].split('_')[0]
                var2 = dir_split[-2].split('_')[0]

                doc_files = [f for f in os.listdir(os.path.join(root,name))]
                for doc_name in doc_files:
                    #print(doc_name,"files")
                    #print("hi")
                    doc_file = os.path.join(root,name)+"/"+doc_name
                    #print(doc_file)
                    f = open(doc_file,"r")
                    get_count(var1,var2,f.read())
                    #print(f.read())

############################CALCULATING PRIOR PROBABILITIES###########################################
def prior_probability():
    c1 = (priors['truthful'])
    c2 = (priors['deceptive'])
    c3 = (priors['positive'])
    c4 = (priors['negative'])

    #print(c1,c2,c3,c4)
    prior_prob['truthful'] = c1 / (c1+c2)
    prior_prob['deceptive'] = c2 / (c1 + c2)

    prior_prob['positive'] = c3 / (c3 + c4)
    prior_prob['negative'] = c4 / (c3 + c4)
    #print(c1+c2,c3+c4)
    #print(priors)
###########################################MAKING THE VOCABULARY###########################################
def total_features():

    features_truthful = class_feat['truthful'].keys()
    [vocabulary.add(f) for f in features_truthful]
    features_deceptive = class_feat['deceptive'].keys()
    [vocabulary.add(f) for f in features_deceptive]

    total_feat['total'] = len(vocabulary)
    #print(total_feat)

#########################################LAPLACE SMOOTHING##############################
def smooth():
    ##laplace smoothing
    #print(class_feat['truthful']['gucci'])
    for x in class_feat:
        for key in class_feat[x]:
            class_feat[x][key]+=1
    #print("************************************************************")
    #print(class_feat['truthful']['gucci'])

    ##above added 1 for each feature

    truthset = set(class_feat['truthful'].keys())
    decpset = set(class_feat['deceptive'].keys())

    notintruth = decpset.difference(truthset)
    notindecp = truthset.difference(decpset)

    for fts in notintruth:
        class_feat['truthful'][fts] = 1

    for fts in notindecp:
        class_feat['deceptive'][fts] = 1

    truthset = set() ##clear memory, not needed
    decpset = set() ##clear memory, not needed

    posset = set(class_feat['positive'].keys())
    negset = set(class_feat['negative'].keys())

    notinpos = negset.difference(posset)
    notinneg = posset.difference(negset)

    for fts in notinpos:
        class_feat['positive'][fts] = 1

    for fts in notinneg:
        class_feat['negative'][fts] = 1

    notinpos = set() ##clear memory, not needed
    notinneg = set() ##clear memory, not needed

    for x in class_feat:
        for y in class_feat[x]:
            class_feat[x][y] = (class_feat[x][y]) / (priors[x] + total_feat['total'])


#######################################MAKING OUTPUT FILE##############################################
def make_output_file():
    final_dict = {'prior_prob':prior_prob,'cond_prob':class_feat}
    model = json.dumps(final_dict,indent = 1)
    f = open(output_file,"w")
    f.write(model)
    f.close()
    #print("output file created")

####################################CALLING ALL FUNCTIONS###################################
read_content()
#print(ip_path)
#print(len(ip_path))
prior_probability()
total_features()
smooth()
make_output_file()
#########################################################################################3
