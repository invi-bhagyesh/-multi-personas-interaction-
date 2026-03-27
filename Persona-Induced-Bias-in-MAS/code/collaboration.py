import os
import json
import argparse
import re
import multiprocessing
from tqdm import tqdm
from prompts import *
from utils import *
from client import send_client

class helper:
    dataset = None
    model = None
    group = None
    num1 = None
    num2 = None
    persona1 = None
    persona2 = None
    turn = None
    initial = None


def start_debate(data):
    agents = []
    for i in range(helper.num1):
        agent = Agent(persona_groups[helper.group][helper.persona1])
        agent.system_instr = personas_prompt(agent.persona)
        agent.task_instr = gpqa_task_prompt("other people", data)
        if helper.initial == 'T':
            agent.output.append(data["correct"][i])
        elif helper.initial == 'F':
            agent.output.append(data["wrong"][i])
        agents.append(agent)
    for i in range(helper.num2):
        agent = Agent(persona_groups[helper.group][helper.persona2])
        agent.system_instr = personas_prompt(agent.persona)
        agent.task_instr = gpqa_task_prompt(agent.persona, data)
        if helper.initial == 'T':
            agent.output.append(data["wrong"][i])
        elif helper.initial == 'F':
            agent.output.append(data["correct"][i])
        agents.append(agent)
    return agents

def start_persuade(data):
    agent_0 = Agent(persona_groups[helper.group][helper.persona1])
    agent_0.system_instr = personas_prompt(agent_0.persona)
    other_people = f"{helper.num2} {persona_(persona_groups[helper.group][helper.persona2])}s"
    agent_0.task_instr = persuadee_prompt(data["claim"], other_people, other_people)
    if helper.initial == 'support' :
        agent_0.output.append(message_prompt(data["support"]) + "<stance> support </stance>")
    elif helper.initial == 'oppose':
        agent_0.output.append(message_prompt(data["oppose"]) + "<stance> oppose </stance>")
    agents = []
    for i in range(helper.num2):
        agent = Agent(persona_groups[helper.group][helper.persona2])
        agent.system_instr = personas_prompt(agent.persona)
        if helper.initial == 'support' :
            agent.task_instr = persuader_prompt(persona_(agent_0.persona), "oppose", data["claim"])
            agent.output.append(message_prompt(data["persuade_support"][i]))
        elif helper.initial == 'oppose':
            agent.task_instr = persuader_prompt(persona_(agent_0.persona), "support", data["claim"])
            agent.output.append(message_prompt(data["persuade_oppose"][i]))
        agents.append(agent)
    return agent_0, agents

def debate(agents):
    interaction = "Considering your answer and other people's answers, can you give an updated answer? You may choose to revise or remain your previous answer. Examine your solution and the solutions of other people step by step. \nExplain your answer, and ensure that your final answer (A) or (B) or (C) or (D) is positioned at the very end of your output inside parentheses, adhering to the format 'Final answer: (answer)'."

    for k in range(len(agents)):
        agent = agents[k]
        memory = []
        memory.append({'role': 'system', 'content': agent.system_instr})
        memory.append({'role': 'user', 'content': agent.task_instr})
        for i in range(len(agent.output)):
            memory.append({'role': 'assistant', 'content': agent.output[i]})
            other_message = ""
            for j in range(len(agents)):
                if j == k:
                    continue
                other_agent = agents[j]
                other_message += gpqa_other_answer(other_agent.persona, other_agent.output[i])
            memory.append({'role': 'user', 'content': other_message + interaction})
        
        answer = send_client(helper.model, memory, 1500)
        agent.output.append(answer)

def persuade(agent, agents):
    memory = []
    memory.append({'role': 'system', 'content': agent.system_instr})
    memory.append({'role': 'user', 'content': agent.task_instr})
    for i in range(len(agent.output)):
        memory.append({'role': 'assistant', 'content': agent.output[i]})
        other_message = ""
        for j in range(len(agents)):
            other_message += interaction_prompt(str(extract_message(agents[j].output[i])))
        memory.append({'role': 'user', 'content': other_message })
    answer = send_client(helper.model, memory, 400)
    agent.output.append(answer)
    if early_stop_persuade(agent):
        return

    for a in agents:
        memory = []
        memory.append({'role': 'system', 'content': a.system_instr})
        memory.append({'role': 'user', 'content': a.task_instr})
        for i in range(len(a.output)):
            memory.append({'role' : 'user', 'content': interaction_prompt(extract_message(agent.output[i]))})
            memory.append({'role': 'assistant', 'content': a.output[i]})
        memory.append({'role': 'user', 'content': interaction_prompt(extract_message(agent.output[-1]))})
        answer = send_client(helper.model, memory, 400)
        if answer == None:
            print(memory)
        a.output.append(answer)

def early_stop_debate(agents):
    answer = extract_option(agents[0].output[-1])
    for agent in agents[1:]:
        if extract_option(agent.output[-1]) != answer:
            return False
    return True
def early_stop_persuade(agent):
    answer = extract_stance(agent.output[-1])
    if not answer:
        return False
    if helper.initial == 'support' and 'oppose' in answer:
        return True
    elif helper.initial == 'oppose' and 'support' in answer:
        return True
    return False

            
def simulate(data,folder_path):
    write_path = os.path.join(folder_path, f"{data["case"]}.json")
    if helper.dataset == 'cps':
        agents = start_debate(data)
        for i in range(helper.turn):
            if early_stop_debate(agents):
                break
            debate(agents)
        with open(write_path, "w") as file:
            json.dump([agent.output for agent in agents], file, indent=4)
        print(data["case"], "done")
    elif helper.dataset == 'persuade':
        agent_0, agents = start_persuade(data)
        for i in range(helper.turn):
            if early_stop_persuade(agent_0):
                break
            persuade(agent_0, agents)
        with open(write_path, "w") as file:
            json.dump({
                "case": data["case"],
                "0": agent_0.output,
                "1": [agent.output for agent in agents]
            }, file, indent=4)
    
def get_results(data,folder_path):
    for i in tqdm(range(len(data))):
        simulate(data[i], folder_path)

def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataset', type=str, default="cps")  
    parser.add_argument('--model', type=str, default="gpt")
    parser.add_argument('--persona1', type=int, default=0)
    parser.add_argument('--persona2', type=int, default=0)
    parser.add_argument('--num1', type=int, default=3)
    parser.add_argument('--num2', type=int, default=3)
    parser.add_argument('--group', type=int, default=1)
    parser.add_argument('--initial', type=str, default='T')
    parser.add_argument('--turn', type=int, default=1)
    return parser.parse_args()

def check(datapath):
    lst = []
    for file in os.listdir(datapath):
        if file.endswith('.json'):
            readpath = os.path.join(datapath, file)
            with open(readpath, 'r') as file:
                data = json.load(file)
                if data[0][-1] == None:
                    #提取文件名中位于/和.json之间全是数字的部分
                    case = re.search(r'\/(\d+)\.json', file.name)
                    lst.append(int(case.group(1))if case else None)
    return lst

def init(args):
    helper.dataset = args.dataset
    helper.model = args.model
    helper.persona1 = args.persona1
    helper.persona2 = args.persona2
    helper.num1 = args.num1
    helper.num2 = args.num2
    helper.group = args.group
    helper.initial = args.initial
    helper.turn = args.turn

def main():
    args = parse_args()
    init(args)
    
    folder_path = os.path.join("results", "collaboration", f"{helper.dataset}/{helper.model}/{helper.group}{helper.persona1}{helper.persona2}_{helper.num1}{helper.num2}_{helper.initial}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if helper.dataset == 'cps':
        data_path = "/home/lijy/Bias/json_data/cps_history.json"
    elif helper.dataset == 'persuade':
        data_path = "/home/lijy/Bias/json_data/persuade_history.json"
    with open(data_path, "r") as file:
        data = json.load(file)
    
    pool = multiprocessing.Pool(16)
    results_unmerged = []
    itv = 20
    for sid in list(range(0, len(data), itv)):
        results_unmerged.append(
            pool.apply_async(get_results, (data[sid:sid+itv],folder_path))
        )
        
    pool.close()
    pool.join()
    


if __name__ == '__main__':
    main()
    