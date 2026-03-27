import os
import json
import argparse
import re
import multiprocessing
import prompts as prompts
import utils as utils
from client import send_client
from tqdm import tqdm

class helper:
    model = None
    group = None

def start(data):
    interaction_prompt = "Answer the following question. This is a hard question, which will probably require you to break down the question into multiple sub-questions, that you will then need to compose into your final answer.{}: A) {}, B) {}, C) {}, D) {} \nExplain your answer, and ensure that your final answer (A) or (B) or (C) or (D) is positioned at the very end of your output inside parentheses, adhering to the format 'Final answer: (answer)'."
    task_prompt = interaction_prompt.format(data["question"], data["options"][0], data["options"][1], data["options"][2], data["options"][3])
    answers = []
    for persona in utils.persona_groups[helper.group]:
        memory = []
        memory.append({"role" : "system" , "content" : utils.personas_prompt(persona)})
        memory.append({"role" : "user" , "content" : task_prompt})
        answer = send_client(helper.model, memory)
        answers.append(answer)
    return answers

def get_result(data):
    results = []
    for idx in tqdm(range(len(data))):
        answers = start(data[idx])
        entry = {
            "case" : data[idx]["case"],
            "answers" : answers
        }
        results.append(entry)
    return results

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--group', type=int, default=0)    
    parser.add_argument('--model', type=str, default="gpt")  
    return parser.parse_args()

def init(args):
    helper.model = args.model
    helper.group = args.group

def main():
    args = parse_args()
    init(args)
    
    
    with open("../data/gpqa_455.json", "r") as file:
        data = json.load(file)
    
    pool = multiprocessing.Pool(32)
    results_unmerged = []
    itv = 15
    for sid in list(range(0, len(data), itv)):
        results_unmerged.append(
            pool.apply_async(get_result, (data[sid:sid+itv],))
        )

    pool.close()
    pool.join()
    results = [r for result in results_unmerged for r in result.get()]
    file_name = "results/accuracy/{}_group{}.json".format(helper.model, helper.group)
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
