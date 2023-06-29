import os
import shutil
from tqdm import tqdm
import json


def name_normal():
    train_and_test = os.listdir('final_dataset')
    for ttv in train_and_test:
        file = os.listdir(os.path.join('final_dataset', ttv))
        for image in file:
            img_list = os.listdir(os.path.join('final_dataset',ttv, image))
            if image == 'mask':
                for i, img_name in enumerate(img_list):
                    new_name = '{:07d}'.format(i)
                    os.rename(os.path.join('final_dataset',ttv, image, img_name), os.path.join('final_dataset',ttv, image, new_name))

            else:
                for i, img_name in enumerate(img_list):
                    img_num, img_last = img_name.split('.')
                    new_name = '{:07d}.{}'.format(i, img_last)
                    os.rename(os.path.join('final_dataset', ttv, image, img_name), os.path.join('final_dataset', ttv, image, new_name))



def generate_json():
    train_and_test = ['testB']#os.listdir('final_dataset')
    for ttv in train_and_test:
        json_filename = "final_dataset/{}/refindoor_{}_with_scene.json".format(ttv, ttv)
        box = os.path.join('final_dataset', ttv, 'box')
        text = os.path.join('final_dataset', ttv, 'text')
        length = len(os.listdir(box))
        info = []
        data_num = 0
        img_num = 0
        for i in range(length):
            box_pt = os.path.join(box, '{:07d}.txt'.format(i))
            text_pt = os.path.join(text, '{:07d}.txt'.format(i))
            image_pt = os.path.join('final_dataset', ttv, 'image', '{:07d}.png'.format(i))
            depth_pt = os.path.join('final_dataset', ttv, 'depth', '{:07d}.png'.format(i))
            with open(box_pt, 'r', encoding='utf-8') as f_box:
                box_name = f_box.readlines()
            with open(text_pt, 'r', encoding='utf-8') as f_text:
                text_name = f_text.readlines()
            mask_pt = os.path.join('final_dataset', ttv, 'mask', '{:07d}'.format(i))
            mask_list = os.listdir(mask_pt)
            box_info = eval(box_name[0])
            #box_info_ = [''.join(filter(str.isdigit, ebox)) for ebox in box_info]
            text_info = eval(text_name[0])
            #text_info_ = [''.join(filter(str.isalpha, etext)) for etext in text_info]
            text_len = len(text_info)
            img_num += 1

            if i < 300:
                for j in range(text_len):
                    info_ij = {'num': data_num, 'text': text_info[j],
                               'bbox': box_info[j], 'rgb_path': image_pt,
                               'depth_path': depth_pt, 'mask_path': os.path.join(mask_pt, mask_list[j]),
                               'scene': 'chair'}
                    print(ttv, i, j, text_len, len(mask_list), len(box_info))

                    info.append(info_ij)
                    data_num += 1
            elif i>=300 and i<481:
                for j in range(text_len):
                    info_ij = {'num': data_num, 'text': text_info[j],
                               'bbox': box_info[j], 'rgb_path': image_pt,
                               'depth_path': depth_pt, 'mask_path': os.path.join(mask_pt, mask_list[j]),
                               'scene': 'shelf'}
                    print(ttv, i, j, text_len, len(mask_list), len(box_info))

                    info.append(info_ij)
                    data_num += 1

            elif i>=481 and i<624:
                for j in range(text_len):
                    info_ij = {'num': data_num, 'text': text_info[j],
                               'bbox': box_info[j], 'rgb_path': image_pt,
                               'depth_path': depth_pt, 'mask_path': os.path.join(mask_pt, mask_list[j]),
                               'scene': 'table'}
                    print(ttv, i, j, text_len, len(mask_list), len(box_info))

                    info.append(info_ij)
                    data_num += 1

            elif i>=624 and i<774:
                for j in range(text_len):
                    info_ij = {'num': data_num, 'text': text_info[j],
                               'bbox': box_info[j], 'rgb_path': image_pt,
                               'depth_path': depth_pt, 'mask_path': os.path.join(mask_pt, mask_list[j]),
                               'scene': 'sofa'}
                    print(ttv, i, j, text_len, len(mask_list), len(box_info))

                    info.append(info_ij)
                    data_num += 1

            elif i>=774 and i<924:
                for j in range(text_len):
                    info_ij = {'num': data_num, 'text': text_info[j],
                               'bbox': box_info[j], 'rgb_path': image_pt,
                               'depth_path': depth_pt, 'mask_path': os.path.join(mask_pt, mask_list[j]),
                               'scene': 'table'}
                    print(ttv, i, j, text_len, len(mask_list), len(box_info))

                    info.append(info_ij)
                    data_num += 1

            elif i>=924 and i<983:
                for j in range(text_len):
                    info_ij = {'num': data_num, 'text': text_info[j],
                               'bbox': box_info[j], 'rgb_path': image_pt,
                               'depth_path': depth_pt, 'mask_path': os.path.join(mask_pt, mask_list[j]),
                               'scene': 'wash table'}
                    print(ttv, i, j, text_len, len(mask_list), len(box_info))

                    info.append(info_ij)
                    data_num += 1

            else:
                for j in range(text_len):
                    info_ij = {'num': data_num, 'text': text_info[j],
                               'bbox': box_info[j], 'rgb_path': image_pt,
                               'depth_path': depth_pt, 'mask_path': os.path.join(mask_pt, mask_list[j]),
                               'scene': 'drawer'}
                    print(ttv, i, j, text_len, len(mask_list), len(box_info))

                    info.append(info_ij)
                    data_num += 1

        print(img_num, data_num)
        with open(json_filename, 'w') as file_obj:
            print(json_filename)
            json.dump(info, file_obj)


if __name__ == '__main__':
    # name_normal()
    generate_json()






