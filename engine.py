# 2022/11/21
import os
import sys
sys.path.append('/home/luyh/vlgrasp/GraspNet/')

from GraspNet.model.FGC_graspnet import FGC_graspnet
from GraspNet.model.decode import pred_decode
from GraspNet.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from GraspNet.utils.collision_detector import ModelFreeCollisionDetector


sys.path.append('/home/luyh/vlgrasp/RoboRefIt/')
from RoboRefIt.models import build_reftr
from RoboRefIt.util import box_ops

from graspnetAPI import GraspGroup
import open3d as o3d
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2



class vl_model():
    def __init__(self, args, device) -> None:
        self.args = args
        self.device = device
        self.checkpoint = args.checkpoint_vl_path
        self.visualize = True
        self.out_dir = args.output_dir
        

    def load_vl_net(self):
        vl_net, criterion, postprocessors = build_reftr(self.args)
        vl_net.half()
        vl_net.to(self.device)
        #criterion.to(self.device)
        
        checkpoint_vl = torch.load(self.checkpoint)
        vl_net.load_state_dict(checkpoint_vl['model'], strict=False)
        start_epoch = checkpoint_vl['epoch']
        print("-> loaded visual grounding checkpoint %s (epoch: %d)"%(self.checkpoint, start_epoch))
        # set vl_net to eval mode
        vl_net.eval()
        #criterion.eval()
        return vl_net, criterion, postprocessors
    
    def process_box(self, bbox, target_size):
        bs, k, _ = bbox.shape

        assert len(bbox) == len(target_size)

        # TODO for multiple predictions
        # print("out_bbox.shape:", out_bbox.shape)
        out_bbox = out_bbox[:, 0, :]
        box = box_ops.box_cxcywh_to_xyxy(out_bbox)
        
        return box
    
    def forward(self, samples):
        vl_net, criterion, postprocessors = self.load_vl_net()
        
        # visualize=False
        if self.visualize:
            output_dir = Path(self.out_dir) / 'vis'
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'mask').mkdir(parents=True, exist_ok=True)
            (output_dir / 'bbox').mkdir(parents=True, exist_ok=True)
            purple = np.array([[[128, 0, 128]]], dtype=np.uint8)
            yellow = np.array([[[255, 255, 0]]], dtype=np.uint8)
            
        from ipdb import set_trace
        #set_trace()
        
        img_ori = samples['img_ori']
        outputs = vl_net(samples)
        
        #set_trace()
        
        bbox, mask = outputs['pred_boxes'], outputs['pred_masks']
        
        # bbox
        bbox = box_ops.box_cxcywh_to_xyxy(bbox[0])
        img_w, img_h = torch.tensor([640]).to(self.device), torch.tensor([480]).to(self.device)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # print(boxes, scale_fct)
        pred_bbox = bbox * scale_fct
        
        # TODO support multi-phrase in the future
        if 'segm' in postprocessors.keys():
            target_sizes = orig_target_sizes = torch.tensor([[480, 640]]).to(self.device)
            results = [{}]
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = results[0]
            pred_mask_ = res['masks'][0][0].cpu().numpy()
            
            if self.visualize:
                pred_mask = res['masks_origin'][0, 0].cpu().unsqueeze(-1).numpy().astype(np.uint8)
                pred_mask = pred_mask * yellow + (1-pred_mask)*purple
                # print(pred_mask.shape, yellow.shape)
                pred_mask = Image.fromarray(pred_mask)
                pred_mask.save(output_dir / 'mask'/ "0.png")

                pred_box = pred_bbox[0][0].detach().cpu().numpy().tolist()
                
                
                #img_bbox = Image.fromarray(img_ori[:,:,::-1])
                img_bbox = Image.fromarray(img_ori)
                draw = ImageDraw.Draw(img_bbox)
                draw.rectangle(pred_box, outline='blue', width=5)

                # cv2.imshow("the visual grounding results", np.hstack((img_bbox, pred_mask)))
                # cv2.waitKeyEx(0)
                # cv2.destroyWindow("the visual grounding results")
                img_bbox.save(output_dir / 'bbox'/ "0.png")
                
                    
        return pred_box, pred_mask_
        

class grasp_model():
    def __init__(self, args, device, image, bbox, mask, text) -> None:
        self.args = args
        
        # input
        self.device = device
        self.img = image
        self.bbox = bbox
        self.text = text
        self.mask = mask
        self.kernel = 0.2
        
        # net parameters
        self.num_view = args.num_view
        self.checkpoint_grasp_path = args.checkpoint_grasp_path
        self.output_path = args.output_dir_grasp
        self.collision_thresh = args.collision_thresh
        
    def load_grasp_net(self):
        # Init the model
        net = FGC_graspnet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax=0.02, is_training=False, is_demo=True)
        
        net.to(self.device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_grasp_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded FGC_GraspNet checkpoint %s (epoch: %d)"%(self.checkpoint_grasp_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net
    
        
    def check_grasp(self, gg):
        # select a suitable pose with high confidence score and ease of manipulation
        gg_30 = GraspGroup()
        gg_60 = GraspGroup()
        gg_90 = GraspGroup()

        score_30, score_60, score_90 = [], [], []
        

        for grasp in gg:
                rot = grasp.rotation_matrix
                translation = grasp.translation
                x, y, z = translation
                score = grasp.score
                #print(score)

                init_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # 相机坐标系先乘初始矩阵到graspnet的初始方向
                rot_predict = np.dot(rot, init_matrix)

                rotate = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) 

                #rotate_predict = np.dot(rotate, rot_)
                distance = np.arccos(1/2*(np.trace(rotate @ rot_predict.T)-1))


                # if distance <= 90 / 180 *np.pi and z < 0.6:
                #     gg_new.add(grasp)
                if distance <= 30 / 180*np.pi:
                    gg_30.add(grasp)
                    score_30.append(score)
                elif distance> 30 / 180*np.pi and distance<= 60 / 180*np.pi:
                    gg_60.add(grasp)
                    score_60.append(score)
                elif distance> 60 / 180*np.pi and distance<= 120 / 180*np.pi:
                    gg_90.add(grasp)
                    score_90.append(score)
        from ipdb import set_trace
        
        if len(score_30 + score_60 + score_90) == 0:
            return GraspGroup()
        ref_value = np.array(score_30 + score_60 + score_90).max()
        ref_min = np.array(score_30 + score_60 + score_90).min()
        score_30 = [x - ref_min for x in score_30]
        score_60 = [x - ref_min for x in score_60]
        score_90 = [x - ref_min for x in score_90]

        factor = 0.4

        if len(gg_30) > 0  and np.array(score_30).max() > ref_value * factor:
            print('select 30')
            return gg_30
        elif len(gg_60) > 0 and np.array(score_60).max() > ref_value * factor:
            print('select 60')
            return gg_60
        else:
            print('select 90')

            return GraspGroup()
            
        
    def pc_to_depth(self, pc, camera):
        x, y, z = pc
        xmap = x*camera.fx / z + camera.cx
        ymap = y*camera.fy / z + camera.cy
        
        return int(xmap), int(ymap)
    
    def choose_in_mask(self, gg):
        camera = CameraInfo(640.0, 480.0, 592.566, 592.566, 319.132, 246.937, 1000)
        gg_new = GraspGroup()
        #print("mask shape", self.mask.shape) mask.shape = 480*640  img.width = 640 img.height = 480
        for grasp in gg:
            rot = grasp.rotation_matrix
            translation = grasp.translation
            if translation[-1] != 0:
                xmap, ymap = self.pc_to_depth(translation, camera)
                #print(xmap, ymap, self.mask[ymap, xmap])
                
                if self.mask[ymap, xmap]:
                    gg_new.add(grasp)
        return gg_new


        
    def get_and_process_data(self, depth):
        # load data
        color = np.array(Image.fromarray(self.img), dtype=np.float32) / 255.0
        
        #workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')).resize((320, 240)))
        #meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        #intrinsic = meta['intrinsic_matrix']
        #factor_depth = meta['factor_depth']
        
        # generate cloud
        '''we use the intrinsic of the Realsense D435i camera in our experiments,
            you can change the intrinsic by yourself.
        '''
        #camera = CameraInfo(640.0, 480.0, 296.282990, 296.282990, 159.566162, 123.468735, 1000)
        #camera = CameraInfo(640.0, 480.0, 386.808, 386.808, 321.153, 240.925, 1000)
        camera = CameraInfo(640.0, 480.0, 592.566, 592.566, 319.132, 246.937, 1000)
        #camera = CameraInfo(640.0, 480.0, 382.2802429199219, 381.8324890136719,  314.146240234375, 243.6321563720703, 1000)
        
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        x1, y1, x2, y2 = map(int, self.bbox)
        x1_, y1_, x2_, y2_ = x1-int((x2-x1)*self.kernel)-50, y1-int((y2-y1)*self.kernel)-50, x2+int((x2-x1)*self.kernel)+50, y2+int((y2-y1)*self.kernel)+50
        xmin, ymin, xmax, ymax = 0, 0, self.mask.shape[1], self.mask.shape[0]
        # xmax = 480, ymax = 640
        
        dx1, dy1, dx2, dy2 = max(x1_, xmin), max(y1_, ymin), min(x2_, xmax), min(y2_, ymax)
        print( x1_, y1_, x2_, y2_, xmin, ymin, xmax, ymax)
        
        mask = np.zeros_like(depth)
        print(mask.shape, depth.shape)
        
        mask[dy1:dy2, dx1:dx2] = 1
        
        mask = mask > 0 & (depth > 0)
        # from ipdb import set_trace
        # set_trace()
        
        
        #mask = (self.mask & (depth > 0))
        
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        print("number of point cloud", len(cloud_masked))
        # sample points
        if len(cloud_masked) >= self.args.num_point:
            idxs = np.random.choice(len(cloud_masked), self.args.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.args.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        #cloud_sampled = xyzrgb[:, 0:3]
        #color_sampled = xyzrgb[:, 3:6]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_sampled.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_sampled.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        
        cloud_sampled = cloud_sampled.to(self.device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled
 
        return end_points, cloud
    
    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        
        return gg_array, gg


    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.args.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.args.collision_thresh)
        gg = gg[~collision_mask]
        return gg


    def vis_grasps(self, gg, cloud):
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        return gg


    def get_top(self, gg):

        result = np.argmax(gg[:, 0])
        chose = gg[result, :]
        chose_xyz = chose[-4:-1]
        chose_rot = np.resize(np.expand_dims(chose[-13:-4], axis=0),(3,3))
        dep = chose[3]
        return chose_xyz, chose_rot, dep
    
    def get_top_gg(self, gg):
        if gg.translations.shape[0] == 0:
            return None, None, None
        xyz = gg.translations[0]
        rot = gg.rotation_matrices[0]
        dep = gg.depths[0]
        return xyz, rot, dep
    
    def forward(self, depth):
        
        grasp_net = self.load_grasp_net()
        end_points, cloud = self.get_and_process_data(depth)
        # from ipdb import set_trace
        # set_trace()
        gg_array, gg = self.get_grasps(grasp_net, end_points)

        # grippers = gg.to_open3d_geometry_list()
        # o3d.visualization.draw_geometries([cloud, *grippers])
        if 'electronic' in self.text:
            gg.sort_by_score()
            gg = self.vis_grasps(gg, cloud)
            xyz, rot, dep = self.get_top_gg(gg)
            return xyz, rot, dep

        gg = self.choose_in_mask(gg)
        

        grippers = gg.to_open3d_geometry_list()
        #o3d.visualization.draw_geometries([cloud, *grippers])
        #gg = self.check_grasp(gg)


        #gg_new = self.check_grasp(gg_array)
        gg = self.collision_detection(gg, np.array(cloud.points))
        # gg = gg.nms()
        gg.sort_by_score()
        gg_array = gg.grasp_group_array

        gg= gg[:1]
        #gg = self.vis_grasps(gg, cloud)
        
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
        np.save(f'{self.output_path}/gg.npy', gg_array)
        o3d.io.write_point_cloud(f'{self.output_path}/cloud.ply', cloud)
        
        #xyz, rot, dep = self.get_top(np.array(gg_array))
        xyz, rot, dep = self.get_top_gg(gg)
        return xyz, rot, dep