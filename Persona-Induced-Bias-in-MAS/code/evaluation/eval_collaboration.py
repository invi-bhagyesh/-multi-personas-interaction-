import os
import json
from utils import *

def extract_debate(folder_path):
    answers = []
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            readpath = os.path.join(folder_path, file)
            with open(readpath, 'r') as file:
                data = json.load(file)
                answer1 = []
                answer2 = []
                for r in data['1']:
                    answer1.append(extract_option(r))
                for r in data['2']:
                    answer2.append(extract_option(r))
                answers.append([answer1, answer2, str(file).split('/')[-1]])
    print(len(answers))
    return answers

def conformity_debate(answers):
    nums = 0
    for answer in answers:
        flag = True
        for i in range(len(answer[0])):
            if answer[0][i] != answer[1][(i+1)%(len(answer[1]))]:
                flag = False
                break
        if flag:
            nums += 1
    print(nums)

    conformity1 = [0,0,0,0,0]
    conformity2 = [0,0,0,0,0]
    conformity3 = [0,0,0,0,0]
    result = [0,0,0]
    for i in answers:
        if i[0][-1] == i[1][-1]:
            if i[0][-1] == i[0][0]:
                conformity1[len(i[0]) - 2] += 1
                result[0] += 1
            elif i[0][-1] == i[1][0]:
                conformity2[len(i[1]) - 2] += 1
                result[1] += 1
            else:
                conformity3[len(i[1]) - 2] += 1
                result[2] += 1
    for i in range(1,5):
        conformity1[i]+= conformity1[i-1]
        conformity2[i]+= conformity2[i-1]
        conformity3[i]+= conformity3[i-1]
    print(result)
    print(conformity1)
    print(conformity2)
    print(conformity3)

def extract_more_debate(folder_path):
    answers = []
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            readpath = os.path.join(folder_path, file)
            with open(readpath, 'r') as file:
                data = json.load(file)
                t1 = []
                for r in data:
                    t2 = []
                    for a in r:
                        t2.append(extract_option(a))   
                    t1.append(t2)
                answers.append(t1)
    
    return answers

def conformity_more_debate(answers):
    conformity1 = [0,0,0,0,0]
    conformity2 = [0,0,0,0,0]
    conformity3 = [0,0,0,0,0]
    
    result = [0,0,0]
    for i in answers:
        flag = True
        answer = i[0][-1]
        for j in range(len(i)):
            if i[j][-1] != answer:
                flag = False
                break
        if flag:
            if answer == i[0][0]:
                conformity1[len(i[0]) - 2] += 1
                result[0] += 1
            elif answer == i[-1][0]:
                conformity2[len(i[-1]) - 2] += 1
                result[1] += 1
            else:
                conformity3[len(i[-1]) - 2] += 1
                result[2] += 1
        else:
            continue
    for i in range(1,5):
        conformity1[i]+= conformity1[i-1]
        conformity2[i]+= conformity2[i-1]
        conformity3[i]+= conformity3[i-1]
    print(conformity1)
    print(conformity2)
    print(conformity3)
    print(result)
    print(len(answers))

def extract_persuade(folder_path):
    stances = []
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            readpath = os.path.join(folder_path, file)
            with open(readpath, 'r') as file:
                data = json.load(file)
                answer = []
                for r in data['0']:
                    answer.append(extract_stance(r))
                stances.append(answer)
    return stances

def conformity_persuade(stances):
    conformity = [0,0,0,0,0,0]
    for i in stances:
        if i[-1] == 'support' and i[0] == 'oppose':
            conformity[len(i)-1] += 1
        elif i[-1] == 'oppose' and i[0] == 'support':
            conformity[len(i)-1] += 1
    print(conformity)
    print(sum(conformity))
    print(len(stances))

answers = extract_more_debate("...")
conformity_more_debate(answers)
stances = extract_persuade("...")
conformity_persuade(stances)



