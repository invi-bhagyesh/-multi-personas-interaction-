import json
import re
from utils import *
from copy import copy
from prompts import persona_groups
import numpy as np


def parse_answer(dataset, readpath):
    with open(readpath,'r') as file:
        data = json.load(file)
    
    with open("json_data/gpqa_TF.json", 'r') as file:
        TF_answers = json.load(file)

    if dataset == "gpqa":
        with open("json_data/gpqa_455.json" , 'r') as file:
            raw = json.load(file)

        TF_rst = []
        FT_rst = []
        initial_answer = []
        for idx in range(len(data)):
            answers = [[],[]]
            TFoutput = data[idx]["correct_face_wrong"]
            FToutput = data[idx]["wrong_face_correct"]
            for o in TFoutput:
                if o == None: answers[0].append(None)
                else: answers[0].append(extract_option(o))
            for o in FToutput:
                if o == None: answers[1].append(None)
                else: answers[1].append(extract_option(o))
            
            TF_rst.append(answers[0])
            FT_rst.append(answers[1])

            initial_answer.append([TF_answers[idx]["correct_option"],TF_answers[idx]["wrong_option"]])

    return TF_rst, FT_rst, initial_answer

def confidence(initial_answer, TF_rst, FT_rst):

    TF_change = []
    FT_change = []
    for idx in range(len(TF_rst[0])):
        TF_change.append(0)
        FT_change.append(0)
        for i in range(len(initial_answer)):
            if TF_rst[i][idx] == initial_answer[i][1]:
                TF_change[idx] +=1
            if FT_rst[i][idx] == initial_answer[i][0]:
                FT_change[idx] +=1
    
    for idx in range(len(TF_rst[0])):
        TF_change[idx] = round(TF_change[idx]/455, 3)
        FT_change[idx] = round(FT_change[idx]/455, 3)
    print(TF_change, FT_change)
    return TF_change, FT_change

def calculate(support, oppose, support_change, oppose_change):
    support_change = np.array(support_change)
    oppose_change = np.array(oppose_change)

    ave_support = np.mean(np.abs(support_change - support))
    ave_oppose = np.mean(np.abs(oppose_change - oppose))

    print("Average T Change: ", ave_support)
    print("Average F Change: ", ave_oppose)
           


datapath = "..."
TF_rst, FT_rst, initial_answer = parse_answer("gpqa", datapath)
TF_change, FT_change = confidence(initial_answer, TF_rst, FT_rst)



