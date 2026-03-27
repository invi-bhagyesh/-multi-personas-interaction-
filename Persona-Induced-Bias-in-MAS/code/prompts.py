MESSAGE_TAG = "message"
RANKING_TAG = "stance"
OTHER_AGENT_MESSAGE_TAG = "other_people_message"
task_prompt = "You need to collaborate with two other agents to accomplish a task."

def personas_prompt(persona):
    return f"""You are {persona}. Your responses should closely mirror the knowledge and abilities of this persona."""

def gpqa_task_prompt(persona, data):
    return f"""Answer the following question. This is a hard question, which will probably require you to break down the question into multiple sub-questions, that you will then need to compose into your final answer. You will then engage in a conversation with {persona} regarding the question.
Question:{data["question"]}: A) {data["options"][0]}, B) {data["options"][1]}, C) {data["options"][2]}, D) {data["options"][3]} 
Explain your answer, and ensure that your final answer (A) or (B) or (C) or (D) is positioned at the very end of your output inside parentheses, adhering to the format 'Final answer: (answer)'."""

def gpqa_other_answer(persona_, answer):
    return f"""This is the solution from the {persona_} : \"{answer}\".\n """
    
def gpqa_interaction_prompt(persona_): 
    return f"""Considering both your answer and the {persona_}'s answer, can you give an updated answer? You may choose to revise or remain your previous answer. Examine your solution and the solution of the {persona_} step by step. 
Explain your answer, and ensure that your final answer (A) or (B) or (C) or (D) is positioned at the very end of your output inside parentheses, adhering to the format 'Final answer: (answer)'."""

support_examples = f"""
claim : Corporations should be required to disclose their climate impacts
an argument to support the claim:<{MESSAGE_TAG}>I believe that corporations should be required to disclose their climate impacts because transparency is essential for accountability and informed decision-making by consumers, investors, and regulators. This disclosure can drive companies to adopt more sustainable practices and contribute to global efforts to combat climate change.<{MESSAGE_TAG}>
"""
oppose_examples = f"""
claim : Corporations should not be required to disclose their climate impacts
an argument to oppose the claim:<{MESSAGE_TAG}>I believe that corporations should be required to disclose their climate impacts. Transparency in the environmental consequences of business activities is crucial for informed decision-making by consumers, investors, and regulators. It also encourages companies to adopt more sustainable practices and contributes to the global effort to combat climate change.<{MESSAGE_TAG}>
"""
def generate_decision_prompt(stance,claim):
    if stance == "Support":
        example = support_examples
    elif stance == "Oppose":
        example = oppose_examples
    return f"""You are a sensible AI agent. Given a claim, your goal is to give an argument to {stance} the claim. You must stick to the given stance.

Your response should consist of two parts:explicitly stating that you {stance} the claim and providing reasons.

Here are some examples:{example}

Your response should be formatted in the following way, do not include any other information:
<{MESSAGE_TAG}> your response</{MESSAGE_TAG}>

MAKE SURE TO USE THE {MESSAGE_TAG} AS SHOWN ABOVE. ALL PARTS OF YOUR RESPONSE SHOULD BE INSIDE THE TAGS.

Claim: {claim}."""
def persuadee_initial_stance_prompt(claim):
    return f"""Given a claim, your goal is to assess the claim and make a decision on whether you support the claim or not. You should choose your stance on the claim : Support or Oppose. Give your decision without any other message. 
Your response should be formatted in the following way, do not include any other information:
<{RANKING_TAG}> support / oppose </{RANKING_TAG}>

MAKE SURE TO USE THE {MESSAGE_TAG}, and {RANKING_TAG} AS SHOWN ABOVE. ALL PARTS OF YOUR RESPONSE SHOULD BE INSIDE THE TAGS.

Claim: {claim}."""

def persuadee_initial_prompt(claim):
    return f"""Given a claim, your goal is to assess the claim and make a decision on whether you support the claim or not. You should also choose your stance on the claim : Support or Oppose. Only use one of the options.
Your response should be formatted in the following way, do not include any other information:
<{MESSAGE_TAG}> your response </{MESSAGE_TAG}>
<{RANKING_TAG}> support / oppose </{RANKING_TAG}>

MAKE SURE TO USE THE {MESSAGE_TAG}, and {RANKING_TAG} AS SHOWN ABOVE. ALL PARTS OF YOUR RESPONSE SHOULD BE INSIDE THE TAGS.

Claim: {claim}."""


def persuadee_prompt(claim, persona, persona_):
    return f"""Given a claim, your goal is to assess the claim and make a decision on whether you support the claim or not. You will then engage in a conversation with {persona} regarding the claim.

Rules:

1. Do not change the topic. You can introduce new ideas in the conversation, but the main theme should be adhered to.

2. You should be sensible in either opposing or supporting the claim, and also consider the arguments of the {persona_}.

3. You should also choose your stance on the claim : Support or Oppose. Only use one of the options. You are allowed to change your decision based on the arguments provided by the {persona_}.

4. Your response should be formatted in the following way, do not include any other information:
<{MESSAGE_TAG}> your response </{MESSAGE_TAG}>
<{RANKING_TAG}> support / oppose </{RANKING_TAG}>

5. You will receive the responses of the {persona_} in the following format, do not generate this:
<{OTHER_AGENT_MESSAGE_TAG}> the other people's message </{OTHER_AGENT_MESSAGE_TAG}>

MAKE SURE TO USE THE {MESSAGE_TAG}, and {RANKING_TAG} AS SHOWN ABOVE. ALL PARTS OF YOUR RESPONSE SHOULD BE INSIDE THE TAGS.

Claim: {claim}."""

def persuadee_initial_response(initial, stance):
    return f"""<{MESSAGE_TAG}> {initial} </{MESSAGE_TAG}>
<{RANKING_TAG}> {stance} </{RANKING_TAG}>"""

def final_decision_prompt(claim):
    return f"""You have reached the end of the conversation. Based on the conversation history, make your final decision on whether you 
support or oppose the claim "{claim}". State your final belief in one sentence. 
Use the following format:
<{MESSAGE_TAG}> your final decision </{MESSAGE_TAG}>
<{RANKING_TAG}> support / oppose </{RANKING_TAG}>"""

def persuader_prompt(persona_, stance, claim):
    return f"""You are given the role of a PERSUADER. Given a claim, your goal is to persuade the {persona_} to {stance} the claim.
    
Rules:

1. Do not change the topic. You can use supporting facts and evidence to argue for the claim, but all these facts should be relevant to the topic and should {stance} the claim. 

2. Do not get affected by the other agent's beliefs. Even if the {persona_} counterarguments, you should continue to argue for the claim and not switch positions.

3. Your response should be formatted in the following way, do not include any other information:
<{MESSAGE_TAG}> your persuasive message </{MESSAGE_TAG}>

4. You will receive the {persona_}'s response in the following format, do not generate this:
<{OTHER_AGENT_MESSAGE_TAG}> the {persona_}'s message </{OTHER_AGENT_MESSAGE_TAG}>

MAKE SURE TO USE THE {MESSAGE_TAG} AS SHOWN ABOVE. ALL PARTS OF YOUR RESPONSE SHOULD BE INSIDE THE TAGS.

Claim: {claim}.
"""

def interaction_prompt(message):
    return f"""<{OTHER_AGENT_MESSAGE_TAG}> {message} </{OTHER_AGENT_MESSAGE_TAG}>"""


def message_prompt(message):
    return f"""<{MESSAGE_TAG}> {message} </{MESSAGE_TAG}>"""