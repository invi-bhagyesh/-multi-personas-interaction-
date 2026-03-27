import re

persona_groups = [["a person"],[ "a woman", "a man", "a transgender man", "a transgender woman", "a non-binary person"], ["a White person", "a Black person", "a Hispanic person", "an Asian person"]]
all_persona = ["a person", "a woman", "a man", "a transgender man", "a transgender woman", "a non-binary person", "a White person", "a Black person", "a Hispanic person", "an Asian person"]
all_persona_ = ["person", "woman", "man", "transgender man", "transgender woman", "non-binary person", "White person", "Black person", "Hispanic person", "Asian person"]

class Agent:
    def __init__(
        self,
        persona:str
    ):
        self.persona = persona
        self.task_instr = None
        self.system_instr = None
        self.output = []
        
def extract_stance(answer):
    if answer == None: return None
    pattern = r"<stance>([\s\S]*?)</stance>"
    match = re.search(pattern, answer)
    if match:
        if 'support' in match.group(1).lower():
            return 'support'
        elif 'oppose' in match.group(1).lower():
            return 'oppose'
        else:
            return None  
    else:
        return None 

def extract_message(answer):
    if answer == None: return None
    pattern = r"<message>([\s\S]*?)</message>"
    match = re.search(pattern, answer)
    if match:
        return match.group(1)  
    else:
        return None 

def extract_option(content):
    if content == None: return None
    
    pattern = r"\((\w+)\)|(\w+)\)"
    match = re.search(r"(?i)final answer(.*)", content)
    if match:
        matches = re.findall(pattern, match.group(1).strip())
        matches = [match[0] or match[1] for match in matches]
        for match_str in matches:
            if match_str.upper() in ["A","B","C","D"]:
                return match_str.upper()

    matches = re.findall(pattern, content)
    matches = [match[0] or match[1] for match in matches]
    for match_str in matches[::-1]:
        if match_str.upper() in ["A","B","C","D"]:
            return match_str.upper()
        else:
            return None

def persona_(persona):
    return re.sub(r'\b(a|an)\s+\b', '', persona)