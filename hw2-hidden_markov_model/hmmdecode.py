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
ans = ""
all_tags = set()
all_words = set()
######################################READING THE MODEL FILE###############################
def read_model_file(model_path):
    with open(model_path,'r',encoding='utf8') as json_file:
        data = json.load(json_file)

        transition_prob = data['transition_prob']
        emission_prob = data['emission_prob']
        #print(emission_prob)
        all_tags = set(list(transition_prob.keys()))
        all_tags.remove("$STARTTAG")

        all_words = set()
        for x in emission_prob:
            #all_words.add(x)
            for y in emission_prob[x]:
                all_words.add(y)

    return transition_prob,emission_prob,all_tags,all_words
############################################READING TEST DATA##############################
def read_test_data(test_path):
    with open(test_path,'r',encoding='utf8') as f:
        test_data = f.readlines()

    data["test_data"] = test_data
##########################################FINDING BEST PROBS#######################
def find_best_prob(tag, probs, transition_prob):

    back = ""
    maxs = -float('inf')

    for tags in probs:

        if maxs < probs[tags] + transition_prob[tags][tag]:
            maxs = probs[tags] + transition_prob[tags][tag]
            back = tags

    return maxs,back
####################################################DECODING######################
def decode(line):

    words = lines.strip().split(" ")

    probs = {0:{}}
    back = {0:{}}

    for tags in all_tags:
        if words[0] in all_words and words[0] in emission_prob[tags]:
            probs[0][tags] = transition_prob['$STARTTAG'][tags] + emission_prob[tags][words[0]]
            back[0][tags] = "$STARTTAG"


        elif words[0] not in all_words:
            probs[0][tags] = transition_prob["$STARTTAG"][tags]
            back[0][tags] = "$STARTTAG"


    for x in range(1,len(words)):
        probs[x] = {}
        back[x] = {}

        for tags in all_tags:
            if words[x] in all_words and words[x] in emission_prob[tags]:
                probs[x][tags], back[x][tags] = find_best_prob(tags, probs[x-1], transition_prob)

                probs[x][tags]+= emission_prob[tags][words[x]]

            elif words[x] not in all_words:
                probs[x][tags], back[x][tags] = find_best_prob(tags, probs[x-1],transition_prob)

    lasttag = ""
    lastprob = -float('inf')

    for tags in probs[len(words)-1]:

        if probs[len(words)-1][tags] + transition_prob[tags]["$ENDTAG"] > lastprob:
            lasttag = tags
            lastprob = probs[len(words)-1][tags] + transition_prob[tags]["$ENDTAG"]


    tagging = []
    pred = lasttag

    for x in range(len(words)-1,-1,-1):
        #print(tagging)
        tagging.append(words[x]+"/"+pred)
        pred = back[x][pred]

    return " ".join(tagging[::-1])
#########################################PREPARING OUTPUT FILE##########################
def prepare_output_file(ans):
    with open(output_file_path,'w',encoding='utf8') as f:
        f.write(ans)
##########################################################BASICS############################
model_path = 'hmmmodel.txt'
output_file_path = 'hmmoutput.txt'
#test_path = 'hmm-training-data/it_isdt_dev_raw.txt'
test_path = sys.argv[1]
a,b,c,d = read_model_file(model_path)
transition_prob = a
emission_prob = b
all_tags = c
all_words = d
read_test_data(test_path)

testingdata = data['test_data']
for lines in testingdata:
    ans+= decode(lines) + "\n"

ans = ans[:len(ans)-1]
prepare_output_file(ans)
