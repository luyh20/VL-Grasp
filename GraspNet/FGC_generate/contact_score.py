from vis_grasp import visObjGrasp, to_open3d_geometry_list
import open3d as o3d
import numpy as np
import os
from graspnetAPI.utils.utils import generate_views
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from tqdm import tqdm
import torch


class Contact_decision():
    def __init__(self, date_root):
        self.date_root = date_root

    def load_grasp_label(self):
        obj_name = list(range(88))
        obj_path = os.path.join(self.date_root, 'grasp_label', '{}_labels.npz'.format(str(0).zfill(3)))
        label = np.load(obj_path)
        points = label['points']
        offsets = label['offsets']
        scores = label['scores']
        collision = label['collision']
        obj_idx = 0
        return points, offsets, scores, collision, obj_idx

    def get_grasp(self):
        sampled_points, offsets, scores, _, obj_idx = self.load_grasp_label()
        num_views, num_angles, num_depths = 300, 12, 4

        point_inds = np.arange(sampled_points.shape[0])
        template_views = generate_views(num_views)
        template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
        template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

        num_points = len(point_inds)
        target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
        target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
        views = np.tile(template_views, [num_points, 1, 1, 1, 1])
        angles = offsets[:, :, :, :, 0]
        depths = offsets[:, :, :, :, 1]
        widths = offsets[:, :, :, :, 2]

        mask1 = (scores > 0)
        mask1_idx = np.where(scores > 0)
        target_points = target_points[mask1]
        views = views[mask1]
        angles = angles[mask1]
        depths = depths[mask1]
        widths = widths[mask1]
        fric_coefs = scores[mask1]

        Rs = batch_viewpoint_params_to_matrix(-views, angles)

        num_grasp = widths.shape[0]
        scores = (1.1 - fric_coefs).reshape(-1, 1)
        widths = widths.reshape(-1, 1)
        heights = 0.02 * np.ones((num_grasp, 1))
        depths = depths.reshape(-1, 1)
        rotations = Rs.reshape((-1, 9))
        object_ids = obj_idx * np.ones((num_grasp, 1), dtype=np.int32)

        obj_grasp_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids]).astype(
            np.float32)

        return sampled_points, obj_grasp_array, mask1_idx

    def cal_dist(self, point1, point2, point3):
        '''

        :param point1: (x1, y1, z1), the point 1 of line
        :param point2: (x2, y2, z2), the point 2 of line
        :param point3: (x3, y3, z3)
        v12 = point1-point2
        v13 = point1-point3
        distance = |v12×v13| / |v12|
        :return: dis

        '''

        vec1 = point1-point2
        vec2 = point1-point3
        dis = abs(torch.linalg.norm(torch.cross(vec1, vec2)))/abs(torch.linalg.norm(vec1))
        dis13_left = torch.linalg.norm(point1-point3)
        dis23_right = torch.linalg.norm(point2-point3)
        if dis13_left <= dis23_right:
            # 0 means point3 close to left contact, 1 means point3 close to right contact
            dis = [dis, 0]
        else:
            dis = [dis, 1]
        return dis

    def init_contact(self, width, depth):
        height = 0.004
        left_point = torch.tensor([depth - height / 2, -width / 2, 0]).cuda()  # 定义抓取接触点的初始点
        right_point = torch.tensor([depth - height / 2, width / 2, 0]).cuda()
        return left_point, right_point

    def contact_find(self):
        obj_pc, grasp, mask1_idx = self.get_grasp()

        obj_pc_t = torch.from_numpy(obj_pc).cuda()
        grasp_t = torch.from_numpy(grasp).cuda()

        grasp_num = grasp_t.shape[0]
        #grasp = grasp[1]

        four_point_all = []
        gravity_score_all = []

        for x in tqdm(range(grasp_num), desc = 'Loading grasp...'):
            grasp_x = grasp_t[x]
            width = grasp_x[1]
            depth = grasp_x[3]
            rot = torch.reshape(grasp_x[4:13], (3, 3))
            center = grasp_x[-4:-1].unsqueeze(1)          # 3*1
            left_point, right_point = self.init_contact(width, depth)

            left_contact = torch.mm(rot, left_point.unsqueeze(1)) + center  # 得到旋转平移后的接触点  3*1
            right_contact = torch.mm(rot, right_point.unsqueeze(1)) + center
            gravity_center = torch.tensor([0, 0, 0]).cuda()

            pc_num = obj_pc_t.shape[0]
            dis = torch.zeros((pc_num, 2)).cuda()
            for i in range(pc_num):
                point3 = obj_pc_t[i]
                dis_i = self.cal_dist(left_contact.squeeze(1), right_contact.squeeze(1), point3)
                dis[i, :] = torch.tensor(dis_i)

            min2max = torch.argsort(dis[:, 0])
            for i in min2max:
                if dis[i, 1]==0:
                    left_idx = i
                    break

            for j in min2max:
                if dis[j, 1]==1:
                    right_idx = j
                    break

            point_target_left = obj_pc_t[left_idx]
            point_target_right = obj_pc_t[right_idx]
            gravity_center_score = self.cal_dist(point_target_left, point_target_right, gravity_center)

            four_point = torch.stack((point_target_left, point_target_right, left_contact.squeeze(1), right_contact.squeeze(1)), axis=0)

            four_point_all.append(four_point)
            gravity_score_all.append(gravity_center_score)
        four_point_all = torch.tensor(four_point_all)
        gravity_score_all = torch.tensor(gravity_score_all)
        return obj_pc_t, grasp, four_point_all, gravity_score_all

    def vis_contact(self):
        obj_pc, grasp, four_point, score_gc = self.contact_find()

        objp3d = o3d.geometry.PointCloud()
        objp3d.points = o3d.utility.Vector3dVector(obj_pc)
        objp3d.paint_uniform_color([0.3, 0.5, 0])

        pc_target = o3d.geometry.PointCloud()
        pc_target.points = o3d.utility.Vector3dVector(four_point)
        pc_target.paint_uniform_color([0, 0, 1])

        gg = to_open3d_geometry_list(grasp)

        o3d.visualization.draw_geometries([*gg, pc_target, objp3d], width=800, height=600, left=50, top=50)


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    root = '../grasp_data'
    contact_decision = Contact_decision(root)
    x, y, z, w = contact_decision.contact_find()