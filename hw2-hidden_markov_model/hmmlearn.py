############################################ALL IMPORTS #########################################
import json
import time
import string
import os
import sys
import math
#########################################ALL VARIABLES###############################
##emision probability 
##transition probability 
emission_prob = {}
transition_prob = {}
data = {}
######################################READING THE TRAINING DATA ###############################
def read_file(training_path):
    #print(training_path)
    with open(training_path,'r',encoding='utf8') as f:
        training_data = f.readlines()

    data["training_data"] = training_data
################################## TRAINING STARTED#########################################
def start_train(data):
    #print(training_data)
    training_data = data["training_data"]
    for lines in training_data:
        #print(lines)
        words = lines.strip().split(" ")
        #print(words)
        #words.insert(0,' /$STARTTAG')
        #words.append(' /$ENDTAG')
        prev_tag = "$STARTTAG"
        for tokens in words:
            #print(tokens)
            wrd, tag = tokens.rsplit("/",1)
            #print(wrd,tag)
            if prev_tag not in transition_prob:
                transition_prob[prev_tag] = {}

            if tag not in transition_prob[prev_tag]:
                transition_prob[prev_tag][tag] = 1

            else:
                transition_prob[prev_tag][tag]+=1

            if tag not in emission_prob:
                emission_prob[tag] = {}

            if wrd not in emission_prob[tag]:
                emission_prob[tag][wrd] = 1

            else:
                emission_prob[tag][wrd]+=1

            prev_tag = tag

        if prev_tag not in transition_prob:
            transition_prob[prev_tag] = {}

        if "$ENDTAG" not in transition_prob[prev_tag]:
            transition_prob[prev_tag]["$ENDTAG"] = 1

        else:
            transition_prob[prev_tag]["$ENDTAG"]+=1

    #print(transition_prob,emission_prob)
##########################################SMOOTHING#############################################
def smoothing(transition_prob,emission_prob):
    all_tags = list(transition_prob.keys()) + ["$ENDTAG"]

    for tags in transition_prob:
        total = 0
        for checking_tags in all_tags:
            if checking_tags not in transition_prob[tags]:
                transition_prob[tags][checking_tags] =1
            else:
                transition_prob[tags][checking_tags]+= 1

            total+=transition_prob[tags][checking_tags]


        for checking_tags in transition_prob[tags]:
            transition_prob[tags][checking_tags] = math.log(transition_prob[tags][checking_tags]/total, 10)


    for tag in emission_prob:
        total = 0

        for wrds in emission_prob[tag]:
            total += emission_prob[tag][wrds]

        for wrds in emission_prob[tag]:
            emission_prob[tag][wrds] = math.log(emission_prob[tag][wrds]/total,10)
######################################PREPARING MODEL OUTPUT FILE################################
def prepare_mode_file(transition_prob, emission_prob):
    output_file = "hmmmodel.txt"
    model = {'transition_prob': transition_prob, 'emission_prob': emission_prob}

    data = json.dumps(model, ensure_ascii = False)
    f = open(output_file,"w",encoding='utf8')
    f.write(data)
    f.close()
########################################## BASICS##########################################
#training_path = "hmm-training-data/it_isdt_train_tagged.txt"
training_path = sys.argv[1]
read_file(training_path)
start_train(data)
smoothing(transition_prob,emission_prob)
prepare_mode_file(transition_prob,emission_prob)
#print(transition_prob,emission_prob)
