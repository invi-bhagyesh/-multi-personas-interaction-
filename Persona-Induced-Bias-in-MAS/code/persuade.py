import os
import json
import argparse
import re
from tqdm import tqdm
from prompts import *
from utils import *
from client import send_client
import multiprocessing

class helper:
    model = None
    group = None
    persona1 = None
    persona2 = None
    initial = None
    turn = None

def start(data, agent1, agent2):
    agent1.system_instr = personas_prompt(agent1.persona)
    agent1.task_instr = persuadee_prompt(data["claim"], agent2.persona, persona_(agent2.persona))
    
    agent2.system_instr = personas_prompt(agent2.persona)

    if helper.initial == 'support':
        agent1.output.append(message_prompt(data["support"]) + "<stance> support </stance>")
        agent2.task_instr = persuader_prompt(persona_(agent1.persona), "oppose", data["claim"])
        agent2.output.append(message_prompt(data["persuade_support"]))
    elif helper.initial == 'oppose':
        agent1.output.append(message_prompt(data["oppose"]) + "<stance> oppose </stance>")
        agent2.task_instr = persuader_prompt(persona_(agent1.persona), "support", data["claim"])
        agent2.output.append(message_prompt(data["persuade_oppose"]))

def persuade(agent1, agent2):
    memory1 = []
    memory1.append({'role': 'system', 'content': agent1.system_instr})
    memory1.append({'role': 'user', 'content': agent1.task_instr})
    memory2 = []
    memory2.append({'role': 'system', 'content': agent2.system_instr})
    memory2.append({'role': 'user', 'content': agent2.task_instr})
    
    assert len(agent1.output) == len(agent2.output)
    for i in range(len(agent1.output)):
        memory1.append({'role': 'assistant', 'content': agent1.output[i]})
        memory1.append({'role': 'user', 'content': interaction_prompt(extract_message(agent2.output[i]))})

        memory2.append({'role': 'user', 'content': interaction_prompt(extract_message(agent1.output[i]))})
        memory2.append({'role': 'assistant', 'content': agent2.output[i]})
    answer1 = send_client(helper.model, memory1, 400)
    agent1.output.append(answer1)
    if early_stop(agent1):
        return
    memory2.append({'role': 'user', 'content': interaction_prompt(extract_message(answer1))})
    answer2 = send_client(helper.model, memory2, 400)
    agent2.output.append(answer2)

def early_stop(agent1):
    answer1 = extract_stance(agent1.output[-1])
    if not answer1:
        return False
    if helper.initial == 'support' and 'oppose' in answer1:
        return True
    elif helper.initial == 'oppose' and 'support' in answer1:
        return True
    return False

def final_decision(agent1, agent2, data):
    memory1 = []
    memory1.append({'role': 'system', 'content': agent1.system_instr})
    memory1.append({'role': 'user', 'content': agent1.task_instr})
    for i in range(len(agent1.output)):
        memory1.append({'role': 'assistant', 'content': agent1.output[i]})
        if i == len(agent1.output) - 1:
            memory1.append({'role': 'user', 'content': interaction_prompt(extract_message(agent2.output[i])) + final_decision_prompt(data["claim"])})
        else:
            memory1.append({'role': 'user', 'content': interaction_prompt(extract_message(agent2.output[i]))})
    answer1 = send_client(helper.model, memory1, 400)
    agent1.output.append(answer1)

def simulate(data, folder_path):
    agent1 = Agent(persona_groups[helper.group][helper.persona1])
    agent2 = Agent(persona_groups[helper.group][helper.persona2])
    start(data, agent1, agent2)

    for i in range(helper.turn):
        persuade(agent1, agent2)
        if early_stop(agent1):
            break
    '''
    write_path = os.path.join(folder_path, f"{data['case']}.json")
    with open(write_path, "w") as file:
        json.dump({
            "case": data["case"],
            "1": agent1.output,
            "2": agent2.output
        }, file, indent=4)
    '''


def get_results(data, folder_path):
    for case in tqdm(data):
        simulate(case, folder_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, default=0)
    parser.add_argument('--model', type=str, default="gpt")
    parser.add_argument('--turn', type=int, default=1)
    parser.add_argument('--persona1', type=int, default=0)
    parser.add_argument('--persona2', type=int, default=0)
    parser.add_argument('--initial', type=str, default='T')
    return parser.parse_args()

def init(args):
    helper.model = args.model
    helper.group = args.group
    helper.turn = args.turn
    helper.persona1 = args.persona1
    helper.persona2 = args.persona2
    helper.initial = args.initial

def main():
    args = parse_args()
    init(args)

    with open("json_data/persuade_cw.json", "r") as file:
        data = json.load(file)
    
    folder_path = f"results/persuade/{helper.model}/{helper.group}_{helper.persona1}{helper.persona2}_{helper.initial}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    pool = multiprocessing.Pool(16)
    results_unmerged = []
    itv = 1
    for sid in list(range(0, len(data[:1]), itv)):
        results_unmerged.append(
            pool.apply_async(get_results, (data[sid:sid+itv],folder_path))
        )
        
    pool.close()
    pool.join()
    

if __name__ == "__main__":
    main()