import json

filename = "template/question.json"


template_v = [{"text": ["pick up the ", "grasp the ", "find the ", "grab the ", "fetch the ", "get the "]}]
template_s = [{"text": ["I need the ", "I want the ", "I want to grasp the ", "I want to get the ", "I want to pick up the ",
                        "I want to grab the ", "I would like to grasp the ", "I would like to get the ", "I would like to pick up the ",
                        "I would like to grab the ", "I need to grasp the ", "I need to get the ", "I need to pick up the ",
                        "I need to grab the ", "I want to grasp the ", "I want to get the ", "I want to pick up the ",
                        "I want to grab the ", "I hope to grasp the ", "I hope to get the ", "I hope to pick up the ",
                        "I hope to grab the "]}]
template_o = [{"text": ["pass me the ", "give me the ", "bring me the ", "fetch me the ", "hand me the "]}]
template_i = [{"text": ["please pass me the ", "please give me the ", "please bring me the ", "please fetch me the ",
                        "please hand me the ", "will you please pass me the ", "will you please give me the ",
                        "will you please bring me the ", "will you please fetch me the ", "will you please hand me the ",
                        "would you please pass me the ", "would you please give me the ", "would you please bring me the ",
                        "would you please fetch me the ", "would you please hand me the ", "could you please pass me the ",
                        "could you please give me the ", "could you please bring me the ",
                        "could you please fetch me the ", "could you please hand me the "]}]
template_q = [{"text": ["where is the ", "could you find the ", "could you give me the ", "could you hand me the ",
                        "could you fetch me the ", "could you pass me the ", "can you find the ", "can you give me the ",
                        "can you hand me the ", "can you fetch me the ", "would you pass me the ",
                        "would you give me the ", "would you fetch me the ", "would you bring me the "]}]


with open(filename, 'w') as file_obj:
    json.dump(template_q, file_obj)