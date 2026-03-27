import json
import re
from utils import *
from copy import copy
from prompts import persona_groups

def parse_answer(dataset, readpath):
    with open(readpath,'r') as file:
        data = json.load(file)
    if dataset == "gpqa":

        all_answers = []
        for idx in range(len(data)):
            answers = []
            output = data[idx]["answers"]
            for o in output:
                if o == None: answers.append(None)
                else: answers.append(extract_option(o))
            
            all_answers.append(answers)
        return all_answers
    
def accuracy(dataset, all_answers):
    if dataset == "gpqa":
        with open("json_data/gpqa_455.json" , 'r') as file:
            raw = json.load(file)
        acc = []
        for i in range(len(all_answers[0])):
            acc.append(0)
            for a in range(len(raw)):
                correct = raw[a]["answer"]
                if all_answers[a][i] == correct:
                    acc[i] += 1

        for i in range(len(acc)):
            acc[i] = round(acc[i]/len(raw), 3)
        print(acc)

readpath = "..."
all_answers = parse_answer("gpqa", readpath)
accuracy("gpqa", all_answers)

