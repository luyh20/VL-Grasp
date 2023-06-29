# 2022/11/21
import os
import argparse
from pathlib import Path
import torch
import random
import numpy as np
from engine import grasp_model, vl_model
from PIL import Image
import cv2
from transformers import BertTokenizerFast, RobertaTokenizerFast
from RoboRefIt.datasets.refer_segmentation import make_refer_seg_transforms


def get_args_parser():
    parser = argparse.ArgumentParser('RefTR For Visual Grounding; FGC-GraspNet For Grasp Pose Detction', add_help=False)
    
    ################################  RefTR For Visual Grounding ##################################
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["img_backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_mask_branch_names', default=['bbox_attention', 'mask_head'], type=str, nargs='+')
    parser.add_argument('--lr_mask_branch_proj', default=1., type=float)
    parser.add_argument('--lr_bert_names', default=["lang_backbone"], type=str, nargs='+')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--warm_up_epoch', default=2, type=int)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--lr_schedule', default='StepLR', type=str)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--ckpt_cycle', default=20, type=int)

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--no_decoder', default=False, action='store_true')
    parser.add_argument('--reftr_type', default='transformer_single_phrase', type=str,
                        help="using bert based reftr vs transformer based reftr")

    # Model parameters
    parser.add_argument('--pretrain_on_coco', default=False, action='store_true')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help="Path to the pretrained model. If set, DETR weight will be used to initilize the network.")
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--ablation', type=str, default='none', help="Ablation")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', default=True,
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--freeze_reftr', action='store_true',
                        help="Train unfreeze reftr for segmentation if the flag is provided")

    # Language model settings
    parser.add_argument('--bert_model', default="bert-base-uncased", type=str,
                        help="bert model name for transformer based reftr")
    parser.add_argument('--img_bert_config', default="./configs/VinVL_VQA_base", type=str,
                        help="For bert based reftr: Path to default image bert ")
    parser.add_argument('--use_encoder_pooler', default=False, action='store_true',
                        help="For bert based reftr: Whether to enable encoder pooler ")
    parser.add_argument('--freeze_bert', action='store_true',
                        help="Whether to freeze language bert")
    parser.add_argument('--max_lang_seq', default=128, type=int,
                        help="Controls maxium number of embeddings in VLTransformer")
    parser.add_argument('--num_queries_per_phrase', default=1, type=int,
                        help="Number of query slots")

    # Loss
    parser.add_argument('--aux_loss', action='store_true',
                        help="Enable auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--use_softmax_ce', action='store_true',
                        help="Whether to use cross entropy loss over all queries")
    parser.add_argument('--bbox_loss_topk', default=1, type=int,
                        help="set > 1 to enbale softmargin loss and topk picking in vg loss ")

    # * Matcher
    # NOTE The coefficient for Matcher better be consistant with the loss 
    # TODO set_cost_class should be 2 when use focal loss from detr
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    # TODO cls_loss_coef should be 2 when use focal loss from detr
    parser.add_argument('--cls_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=1, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset', default='roborefit')
    parser.add_argument('--data_root', default='data/final_dataset')
    parser.add_argument('--train_split', default='train')
    parser.add_argument('--test_split', default='testA', type=str)
    parser.add_argument('--img_size', default=640, type=int)
    parser.add_argument('--img_type', default='RGB')
    parser.add_argument('--max_img_size', default=640, type=int)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/mscoco', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    
    # demo input
    parser.add_argument('--image_path', default='example/image')
    parser.add_argument('--depth_path', default='example/depth')
    parser.add_argument('--text_path', default='example/text')
    
    
    parser.add_argument('--checkpoint_vl_path', default='logs/checkpoint_best_r50.pth')
    
    parser.add_argument('--output_dir', default='outputs/roborefit',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--resume_model_only', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--run_epoch', default=500, type=int, metavar='N',
                        help='epochs for current run')                
    parser.add_argument('--eval', default=False)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    
    
    ############################ FGC-GraspNet For Grasp Pose Detction ##############################
    parser.add_argument('--checkpoint_grasp_path', default='logs/checkpoint_fgc.tar', help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=12000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
    parser.add_argument('--output_dir_grasp', default='outputs/graspnet')
    
    return parser
        

def build_bert_tokenizer(bert_model):
    if bert_model.split('-')[0] == 'roberta':
        lang_backbone = RobertaTokenizerFast.from_pretrained(bert_model, do_lower_case=True, do_basic_tokenize=False)
    else:
        lang_backbone = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=True, do_basic_tokenize=False)
    return lang_backbone
        
    
    
def main(args):
    # training setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    
    # load the image and depth
    img_path = os.path.join(args.image_path, '0003000.png')
    depth_path = os.path.join(args.depth_path, '0003000.png')
    text_path = os.path.join(args.text_path, '3000.txt')
    
    img_ori = cv2.imread(img_path)
    depth_ori = np.array(Image.open(depth_path))
    with open(text_path, 'r') as file:
        text = file.read()
    
    # process the image
    transform_img = make_refer_seg_transforms(args.img_size, args.max_img_size ,test=True, img_type='RGB')
    
    if img_ori.shape[-1] > 1:
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    else:
        img_ori = np.stack([img_ori] * 3)
    
    img = Image.fromarray(img_ori)
    img, target = transform_img(img, target=None)
    img = img.unsqueeze(0)
    
    
    # encode the sentence
    text = text.lower()
    tokenizer = build_bert_tokenizer(bert_model='bert-base-uncased')
    tokenized_sentence = tokenizer(
        text,
        padding='max_length',
        max_length=30,
        truncation=True,
        return_tensors='pt',
    )
    word_id = tokenized_sentence['input_ids'][0]
    word_mask = tokenized_sentence['attention_mask'][0]
    
    word_id = word_id.unsqueeze(0)
    word_mask = word_mask.unsqueeze(0)
    

    samples = {
        "img": img.to(device).half(),
        "sentence": word_id.to(device),
        "sentence_mask": word_mask.to(device).half(),
        "img_ori": img_ori
    }
    
    
    # load the visual grounding vl_net
    vl_net = vl_model(args, device)
    bbox, mask = vl_net.forward(samples)

    
    # image crop 
    
    # load the grasp pose detection vl_net
    grasp_net = grasp_model(args, device, img_ori, bbox, mask, text)
    xyz, rot, dep = grasp_net.forward(depth_ori)
    
    return xyz, rot, dep
    



if __name__=='__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


    
    