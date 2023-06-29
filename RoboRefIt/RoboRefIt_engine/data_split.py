import os
import shutil
from tqdm import tqdm
import random
random.seed(100)


def moveFile(fileDir, tarDir, num_name, path_num):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)

    rate = 0.85  # the sample rate, devide one scene to train/test
    picknumber = int(filenumber * rate)

    print(picknumber)

    for name in path_num[0:picknumber]:
        name_out = num_name + pathDir[name]
        shutil.copyfile(os.path.join(fileDir, pathDir[name]), tarDir[0] + name_out)

    for name3 in path_num[picknumber:filenumber]:
        name_out = num_name + pathDir[name3]
        shutil.copyfile(os.path.join(fileDir, pathDir[name3]), tarDir[1] + name_out)


def moveFile_all(fileDir, tarDir, num_name, path_num):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)

    for name in path_num:
        name_out = num_name + pathDir[name]
        shutil.copyfile(os.path.join(fileDir, pathDir[name]), tarDir + name_out)


def moveFiletree_all(fileDir, tarDir, num_name, path_num):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)

    for name in path_num:
        name_out = num_name + pathDir[name]
        shutil.copytree(os.path.join(fileDir, pathDir[name]), tarDir + name_out)


def moveFiletree(fileDir, tarDir, num_name, path_num):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)

    rate = 0.85
    picknumber = int(filenumber * rate)
    print(picknumber)

    for name in path_num[0:picknumber]:
        name_out = num_name + pathDir[name]
        shutil.copytree(os.path.join(fileDir, pathDir[name]), tarDir[0] + name_out)

    for name3 in path_num[picknumber:filenumber]:
        name_out = num_name + pathDir[name3]
        shutil.copytree(os.path.join(fileDir, pathDir[name3]), tarDir[1] + name_out)


def generate_train(file_name='final_dataset'):
    train_list = ['train', 'testA', 'testB']
    f_list = ['image', 'mask', 'box', 'text', 'depth', 'depth_colormap']
    for i in train_list:
        for j in f_list:
            x_name = os.path.join(file_name, i, j)
            if not os.path.exists(x_name):
                os.makedirs(x_name)


def pipeline(file_name='4623'):
    scene_list = os.listdir(file_name)
    for num1, file in enumerate(scene_list):
        file_path = os.path.join(file_name, file)
        scene_num = os.listdir(file_path)
        scene_len = len(scene_num)
        for num2, scene in enumerate(scene_num):
            scene_path = os.path.join(file_path, scene)
            num_out = '{}{:02d}'.format(num1+4, num2)

            img_source = os.path.join(scene_path, 'color')

            pathDir = os.listdir(img_source)  
            filenumber = len(pathDir)
            path_num = list(range(filenumber))
            random.shuffle(path_num)
            if num2 > len(scene_num)-4:
                img_target = ['final_dataset/train/image/', 'final_dataset/testB/image/'][1]
                moveFile_all(img_source, img_target, num_out, path_num)

                mask_source = os.path.join(scene_path, 'mask')
                mask_target = ['final_dataset/train/mask/', 'final_dataset/testB/mask/'][1]
                moveFiletree_all(mask_source, mask_target, num_out, path_num)

                box_source = os.path.join(scene_path, 'bbox')
                box_target = ['final_dataset/train/box/', 'final_dataset/testB/box/'][1]
                moveFile_all(box_source, box_target, num_out, path_num)

                text_source = os.path.join(scene_path, 'expression')
                text_target = ['final_dataset/train/text/', 'final_dataset/testB/text/'][1]
                moveFile_all(text_source, text_target, num_out, path_num)

                depth_source = os.path.join(scene_path, 'depth')
                depth_target = ['final_dataset/train/depth/', 'final_dataset/testB/depth/'][1]
                moveFile_all(depth_source, depth_target, num_out, path_num)

                depth_color_source = os.path.join(scene_path, 'depth_colormap')
                depth_color_target = ['final_dataset/train/depth_colormap/', 'final_dataset/testB/depth_colormap/'][1]
                moveFile_all(depth_color_source, depth_color_target, num_out, path_num)
            elif num2 / 4 == 0:
                img_target = ['final_dataset/train/image/', 'final_dataset/testA/image/'][1]
                moveFile_all(img_source, img_target, num_out, path_num)

                mask_source = os.path.join(scene_path, 'mask')
                mask_target = ['final_dataset/train/mask/', 'final_dataset/testA/mask/'][1]
                moveFiletree_all(mask_source, mask_target, num_out, path_num)

                box_source = os.path.join(scene_path, 'bbox')
                box_target = ['final_dataset/train/box/', 'final_dataset/testA/box/'][1]
                moveFile_all(box_source, box_target, num_out, path_num)

                text_source = os.path.join(scene_path, 'expression')
                text_target = ['final_dataset/train/text/', 'final_dataset/testA/text/'][1]
                moveFile_all(text_source, text_target, num_out, path_num)

                depth_source = os.path.join(scene_path, 'depth')
                depth_target = ['final_dataset/train/depth/', 'final_dataset/testA/depth/'][1]
                moveFile_all(depth_source, depth_target, num_out, path_num)

                depth_color_source = os.path.join(scene_path, 'depth_colormap')
                depth_color_target = ['final_dataset/train/depth_colormap/', 'final_dataset/testA/depth_colormap/'][1]
                moveFile_all(depth_color_source, depth_color_target, num_out, path_num)
            else:
                img_target = ['final_dataset/train/image/', 'final_dataset/testA/image/']
                moveFile(img_source, img_target, num_out, path_num)

                mask_source = os.path.join(scene_path, 'mask')
                mask_target = ['final_dataset/train/mask/', 'final_dataset/testA/mask/']
                moveFiletree(mask_source, mask_target, num_out, path_num)

                box_source = os.path.join(scene_path, 'bbox')
                box_target = ['final_dataset/train/box/', 'final_dataset/testA/box/']
                moveFile(box_source, box_target, num_out, path_num)

                text_source = os.path.join(scene_path, 'expression')
                text_target = ['final_dataset/train/text/', 'final_dataset/testA/text/']
                moveFile(text_source, text_target, num_out, path_num)

                depth_source = os.path.join(scene_path, 'depth')
                depth_target = ['final_dataset/train/depth/', 'final_dataset/testA/depth/']
                moveFile(depth_source, depth_target, num_out, path_num)

                depth_color_source = os.path.join(scene_path, 'depth_colormap')
                depth_color_target = ['final_dataset/train/depth_colormap/', 'final_dataset/testA/depth_colormap/']
                moveFile(depth_color_source, depth_color_target, num_out, path_num)
            # img_target = ['final_dataset/train/image/', 'final_dataset/test/image/']
            # moveFile(img_source, img_target, num_out, path_num
            #
            # mask_source = os.path.join(scene_path, 'mask')
            # mask_target = ['final_dataset/train/mask/', 'final_dataset/test/mask/']
            # moveFiletree(mask_source, mask_target, num_out, path_num)
            #
            # box_source = os.path.join(scene_path, 'bbox')
            # box_target = ['final_dataset/train/box/', 'final_dataset/test/box/']
            # moveFile(box_source, box_target, num_out, path_num)
            #
            # text_source = os.path.join(scene_path, 'expression')
            # text_target = ['final_dataset/train/text/', 'final_dataset/test/text/']
            # moveFile(text_source, text_target, num_out, path_num)
            #
            # depth_source = os.path.join(scene_path, 'depth')
            # depth_target = ['final_dataset/train/depth/', 'final_dataset/test/depth/']
            # moveFile(depth_source, depth_target, num_out, path_num)
            #
            # depth_color_source = os.path.join(scene_path, 'depth_colormap')
            # depth_color_target = ['final_dataset/train/depth_colormap/', 'final_dataset/test/depth_colormap/']
            # moveFile(depth_color_source, depth_color_target, num_out, path_num)



if __name__ == '__main__':
    generate_train()
    pipeline()



