import argparse
import json
import os
import random
random.seed(10)


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--template_dir', default='template', help="Directory containing JSON templates")
    parse.add_argument('--img_data', default='scene_robotic_chair2', help="Directory containing image data")
    parse.add_argument('--text_per_img', default=5, type=int)
    parse.add_argument('--text_file', default='expression', help="the text file under the scene_xx")

    return parse


def read_template(args):
    num_loaded_templates = 0
    templates = {}
    for fn in os.listdir(args.template_dir):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(args.template_dir, fn), 'r') as f:
            base = os.path.splitext(fn)[0]
            for i, template in enumerate(json.load(f)):
                num_loaded_templates += 1
                key = base
                templates[key] = template
    print('Read %d templates from disk' % num_loaded_templates)
    return templates


def split_text(input_text):
    '''

    :param input_text: str
    :return: obj name: list
    '''
    input_text = input_text.strip('\n')
    obj_group = input_text.split(':')[-1]
    obj_name = obj_group.split('/')
    return obj_name


def lr_position(input_position):
    pos_dict1 = {'l': 'on the left', 'r': 'on the right'}
    out_pos = pos_dict1[input_position]
    return out_pos


def split_text2(input_text):
    '''

    :param input_text: str
    :return: obj name: list
    '''
    input_text = input_text.strip('\n')
    obj_group = input_text.split(':')[-1]
    obj_name, obj_position = obj_group.split('+')
    obj_name = obj_name.strip().split('/')
    obj_position = obj_position.strip()
    obj_pos = lr_position(obj_position)
    return obj_name, obj_pos


def select_template(templates, text_num):
    '''

    :param templates:  type=dict, {imperative:{'text': [list]}, ... }
    :param text_num: type=int max=5
    :return:
    '''

    temps = []
    choice_num = {'imperative': 3, 'object': 1, 'question': 2, 'subject': 3, 'verb': 1}
    for key, value in templates.items():
        temp = value['text']
        chosen_num = choice_num[key]
        temp_choice = random.sample(temp, chosen_num)
        temps.extend(temp_choice)
    temps_ = random.sample(temps, text_num)
    return temps_


def text_synthesis(obj_name, res_template):
    '''

    :param obj_name: single object name list
    :param res_template: single template
    :return: sentence: str
    '''
    obj_chose = random.choice(obj_name)
    obj_chose_ = obj_chose.replace('_', ' ')
    sentence = res_template + obj_chose_
    return sentence


def text_synthesis2(obj_name, res_template, position):
    '''

    :param obj_name: single object name list
    :param res_template: single template
    :return: sentence: str
    '''
    obj_chose = random.choice(obj_name)
    obj_chose_ = obj_chose.replace('_', ' ')
    sentence = res_template + obj_chose_ + ' ' + position
    return sentence


def template_fill(args, templates, scene_info):
    '''
    templates: type=dict, {imperative:{'text': [list]}, ... }
    scene_info:  type=list, length=2n-1
    '''
    obj_num = len(scene_info)
    max_text_num = args.text_per_img
    text_num = min(obj_num, max_text_num)

    res_template = select_template(templates, obj_num)

    # each object will chose one obj name and combined with one template from res_template
    sentences = []
    for i in range(obj_num):
        obj_info = scene_info[i]
        if '+' in obj_info:
            obj_name, position = split_text2(obj_info)
            sentence = text_synthesis2(obj_name, res_template[i], position)
        else:
            obj_name = split_text(obj_info)
            sentence = text_synthesis(obj_name, res_template[i])
        sentences.append(sentence)
    sentences = random.sample(sentences, text_num)
    return sentences


def main(args):
    templates = read_template(args)

    scenes = os.listdir(args.img_data)
    if 'ls.txt' in scenes:
        scenes.remove('ls.txt')

    # scene_len = len(scenes)
    # scene_id = range(scene_len)
    for i, scene_id in enumerate(scenes):
        scene_path = os.path.join(args.img_data, scene_id)
        img_per_scene = os.listdir(os.path.join(scene_path, 'color'))
        img_num = len(img_per_scene)
        cur_file = os.listdir(scene_path)
        # mkdir scene_xx/expression/
        scene_text_path = os.path.join(scene_path, args.text_file)
        if not os.path.exists(scene_text_path):
            os.mkdir(scene_text_path)

        for x in cur_file:
            if x.startswith('obj'):
                scene_info_path = os.path.join(scene_path, x)
                with open(scene_info_path, 'r', encoding='utf-8') as f_info:
                    scene_info = f_info.readlines()
                    for line in scene_info:
                        if line=='\n':
                            scene_info.remove(line)

                # storage img_id.txt under expression
                for j in range(img_num):
                    expression_per_scene = template_fill(args, templates, scene_info)
                    txt_filename = os.path.join(scene_text_path, '{:02d}.txt'.format(j))
                    with open(txt_filename, "w") as f_txt:
                        # for line in expression_per_scene:
                        #     f_txt.write(line+'\n')
                        f_txt.write(str(expression_per_scene))
                        f_txt.close()


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    #read_template(args)
    main(args)
