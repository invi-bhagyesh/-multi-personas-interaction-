import json
import re
import numpy as np
from utils import *

def conformity(datapath):
    with open(datapath, "r") as file:
        data = json.load(file)
    support = []
    oppose = []
    for i in range(len(data[0]["given_support"])):
        support.append(0)
        oppose.append(0)
        for case in data:
            if extract_stance(case["given_support"][i]) and "oppose" in extract_stance(case["given_support"][i]):
                support[i] += 1
            if extract_stance(case["given_oppose"][i]) and "support" in extract_stance(case["given_oppose"][i]):
                oppose[i] += 1
    support = [round(x / len(data),3) for x in support]
    oppose = [round(x / len(data),3) for x in oppose]
    print("Support Conformity: ", support)
    print("Oppose Conformity: ", oppose)
    return support, oppose

def stance_conformity(stance_path, datapath):
    with open(stance_path, "r") as file:
        stance = json.load(file)
    initial_support = []
    initial_oppose = []
    with open("results/claims/initial_gpt_group2.json", "r") as file:
        data = json.load(file)
    for i in range(len(data[0])):
        if data[0][i] == data[1][i] and data[1][i] == data[2][i] and data[2][i] == data[3][i] and data[3][i] == 1:
            initial_support.append(i)
        if data[0][i] == data[1][i] and data[1][i] == data[2][i] and data[2][i] == data[3][i] and data[3][i] == 0:
            initial_oppose.append(i)
    print(len(initial_support))
    print(len(initial_oppose))
    same_nums = len(initial_support) + len(initial_oppose)
   
    with open(datapath, "r") as file:
        data = json.load(file)
    support = [[],[]]
    oppose = [[],[]]
    same_stance = []
    diff_stance = []
    for i in range(len(data[0]["given_support"])):
        support[0].append(0)
        support[1].append(0)
        oppose[0].append(0)
        oppose[1].append(0)
        same_stance.append(0)
        diff_stance.append(0)
        for j in initial_support:
            if extract_stance(data[j]["given_support"][i]) and "oppose" in extract_stance(data[j]["given_support"][i]):
                support[0][i] += 1
                same_stance[i] += 1
            if extract_stance(data[j]["given_oppose"][i]) and "support" in extract_stance(data[j]["given_oppose"][i]):
                support[1][i] += 1
                diff_stance[i] += 1
        for j in initial_oppose:
            if extract_stance(data[j]["given_support"][i]) and "oppose" in extract_stance(data[j]["given_support"][i]):
                oppose[0][i] += 1
                diff_stance[i] += 1
            if extract_stance(data[j]["given_oppose"][i]) and "support" in extract_stance(data[j]["given_oppose"][i]):
                oppose[1][i] += 1
                same_stance[i] += 1
    
    
    support[0] = [round(x / len(initial_support),3) for x in support[0]]
    support[1] = [round(x / len(initial_support),3) for x in support[1]]
    oppose[0] = [round(x / len(initial_oppose),3) for x in oppose[0]]
    oppose[1] = [round(x / len(initial_oppose),3) for x in oppose[1]]
    same_stance = [round(x / same_nums,3) for x in same_stance]
    diff_stance = [round(x / same_nums,3) for x in diff_stance]

    print("Initial Support give support: ", support[0])
    print("Initial Support give oppose: ", support[1])
    print("Initial Oppose give support: ", oppose[0])
    print("Initial Oppose give oppose: ", oppose[1])
    print("Same stance: ", same_stance)
    print("Different stance: ", diff_stance)

def calculate(support, oppose, support_change, oppose_change):
    support_change = np.array(support_change)
    oppose_change = np.array(oppose_change)

    ave_support = np.mean(np.abs(support_change - support))
    ave_oppose = np.mean(np.abs(oppose_change - oppose))

    print("Average Support Change: ", ave_support)
    print("Average Oppose Change: ", ave_oppose)

support, oppose = conformity("...")