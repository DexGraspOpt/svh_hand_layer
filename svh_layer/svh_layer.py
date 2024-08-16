# leap_hand layer for torch
import torch
import trimesh
import os
import numpy as np
import copy
import pytorch_kinematics as pk


import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layer_asset_utils import save_part_mesh, sample_points_on_mesh, sample_visible_points

# All lengths are in mm and rotations in radians


class SvhHandLayer(torch.nn.Module):
    def __init__(self, to_mano_frame=True, show_mesh=False, hand_type='right', device='cuda'):
        super().__init__()
        self.BASE_DIR = os.path.split(os.path.abspath(__file__))[0]
        self.show_mesh = show_mesh
        self.to_mano_frame = to_mano_frame
        self.device = device
        self.name = 'svh_hand'
        self.hand_type = hand_type
        self.finger_num = 5

        urdf_path = os.path.join(self.BASE_DIR, '../assets/schunk_svh_hand_{}.urdf'.format(hand_type))
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=device)

        # get mimic joint
        activate_joints = list(range(self.chain.n_joints))
        self.mimic_joints = []
        self.mimic_joints_info = []

        self.robot = pk.urdf.URDF.from_xml_file(urdf_path)
        for joint in self.robot.joints:
            if joint.type == 'fixed':
                continue
            if joint.mimic != None:
                index = self.chain.get_joint_parameter_names().index(joint.name)
                mimic_index = self.chain.get_joint_parameter_names().index(joint.mimic.joint)
                self.mimic_joints.append(index)
                self.mimic_joints_info.append((index, mimic_index, joint.mimic.multiplier, joint.mimic.offset, joint.name, joint.mimic.joint))

        if len(self.mimic_joints) > 0:
            self.activate_joints = list(set(activate_joints) - set(self.mimic_joints))
            # self.reorder_mimic_joints = [(activate_joints.index(idx[0]), *idx[1:])for idx in mimic_indices]

        self.joints_lower = self.chain.low[self.activate_joints]
        self.joints_upper = self.chain.high[self.activate_joints]
        self.joints_mean = (self.joints_lower + self.joints_upper) / 2
        self.joints_range = self.joints_mean - self.joints_lower
        self.joint_names = self.chain.get_joint_parameter_names()
        self.n_dofs = self.chain.n_joints - len(self.mimic_joints)

        # self.link_dict = {}
        # for link in self.chain.get_links():
        #     self.link_dict[link.name] = link.visuals[0].geom_param[0].split('/')[-1]
        #     self.scale = link.visuals[0].geom_param[1]

        # order in palm -> thumb -> index -> middle -> ring [-> pinky(little)]
        self.order_keys = [
            'right_hand_base_link', 'right_hand_e1', 'right_hand_e2',  # palm
            'right_hand_z', 'right_hand_a', 'right_hand_b', 'right_hand_c',  # thumb
            'right_hand_virtual_l', 'right_hand_l', 'right_hand_p', 'right_hand_t',  # index
            'right_hand_virtual_k', 'right_hand_k', 'right_hand_o', 'right_hand_s',  # middle
            'right_hand_virtual_j', 'right_hand_j', 'right_hand_n', 'right_hand_r',  # ring
            'right_hand_virtual_i', 'right_hand_i', 'right_hand_m', 'right_hand_q',  # little
        ]

        self.ordered_finger_endeffort = ['right_hand_e2',  'right_hand_c', 'right_hand_t', 'right_hand_s', 'right_hand_r', 'right_hand_q']

        # transformation for align the robot hand to mano hand frame, used for
        self.to_mano_transform = torch.eye(4).to(torch.float32).to(device)
        if self.to_mano_frame:
            import roma
            self.to_mano_transform[:3, :] = torch.tensor([[0.0, 0.0, -1.0, 0.128],
                                                         [0.0, -1.0, 0.0, -0.0075],
                                                         [-1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

            tmp_pose = torch.eye(4, dtype=torch.float32).to(device)
            tmp_pose[:3, :3] = roma.rotvec_to_rotmat(torch.tensor([0., -0.175, 0.0], dtype=torch.float32))
            self.to_mano_transform = torch.matmul(tmp_pose, self.to_mano_transform)

        self.register_buffer('base_2_world', self.to_mano_transform)

        if not (os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_meshes_cvx')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_points')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_composite_points')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/visible_point_indices')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand.obj')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_all_zero.obj')
        ):
            # for first time run to generate contact points on the hand, set the self.make_contact_points=True
            self.make_contact_points = True
            self.create_assets()
        else:
            self.make_contact_points = False

        self.meshes = self.load_meshes()
        self.hand_segment_indices, self.hand_finger_indices = self.get_hand_segment_indices()

    def create_assets(self):
        '''
        To create needed assets for the first running.
        Should run before first use.
        '''
        self.to_mano_transform = torch.eye(4).to(torch.float32).to(device)
        pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
        theta = np.zeros((1, self.n_dofs), dtype=np.float32)

        save_part_mesh()
        sample_points_on_mesh()

        show_mesh = self.show_mesh
        self.show_mesh = True
        self.make_contact_points = True

        self.meshes = self.load_meshes()

        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        parts = mesh.split()

        new_mesh = trimesh.boolean.boolean_manifold(parts, 'union')
        new_mesh.export(os.path.join(self.BASE_DIR, '../assets/hand.obj'))

        self.show_mesh = True
        self.make_contact_points = False
        self.meshes = self.load_meshes()
        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        mesh.export(os.path.join(self.BASE_DIR, '../assets/hand_all_zero.obj'))

        self.show_mesh = False
        self.make_contact_points = True
        self.meshes = self.load_meshes()

        self.get_forward_vertices(pose, theta)      # SAMPLE hand_composite_points
        sample_visible_points()

        self.show_mesh = True
        self.make_contact_points = False
        import roma
        self.to_mano_transform[:3, :] = torch.tensor([[0.0, 0.0, -1.0, 0.128],
                                                      [0.0, -1.0, 0.0, -0.0075],
                                                      [-1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        tmp_pose = torch.eye(4, dtype=torch.float32).to(device)
        tmp_pose[:3, :3] = roma.rotvec_to_rotmat(torch.tensor([0., -0.175, 0.0], dtype=torch.float32))
        self.to_mano_transform = torch.matmul(tmp_pose, self.to_mano_transform)

        self.meshes = self.load_meshes()
        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        mesh.export(os.path.join(self.BASE_DIR, '../assets/hand_to_mano_frame.obj'))

        self.make_contact_points = False
        self.show_mesh = show_mesh

    def load_meshes(self):
        meshes = {}
        for link in self.chain.get_links():
            link_name = link.name
            if link_name not in self.order_keys:
                continue
            for visual in link.visuals:
                if visual.geom_type == None:
                    continue
                if visual.geom_type == 'mesh':
                    rel_path = visual.geom_param[0]
                    mesh_filepath = os.path.abspath(os.path.join(self.BASE_DIR, '../assets/', rel_path))
                    assert os.path.exists(mesh_filepath)
                    scale = visual.geom_param[1] if visual.geom_param[1] else [1.0, 1.0, 1.0]
                    link_pre_transform = visual.offset
                    mesh = trimesh.load(mesh_filepath, force='mesh')
                elif visual.geom_type == 'box':
                    mesh = trimesh.creation.box(extents=visual.geom_param)
                    scale = [1.0, 1.0, 1.0]
                    mesh_filepath = None
                else:
                    raise NotImplementedError

                if self.show_mesh:
                    if self.make_contact_points and mesh_filepath is not None:
                        mesh_filepath = mesh_filepath.replace('assets/hand_meshes/', 'assets/hand_meshes_cvx/').replace('.obj', '.stl')
                        mesh = trimesh.load(mesh_filepath, force='mesh')
                    mesh.apply_scale(scale)

                    verts = link_pre_transform.transform_points(torch.FloatTensor(np.array(mesh.vertices)))

                    temp = torch.ones(mesh.vertices.shape[0], 1).float()
                    vertex_normals = link_pre_transform.transform_normals(
                        torch.FloatTensor(copy.deepcopy(mesh.vertex_normals)))

                    meshes[link_name] = [
                        torch.cat((verts, temp), dim=-1).to(self.device),
                        mesh.faces,
                        torch.cat((vertex_normals, temp), dim=-1).to(self.device).to(torch.float)
                    ]
                else:
                    if mesh_filepath is not None:
                        vertex_path = mesh_filepath.replace('hand_meshes', 'hand_points').replace('.stl', '.npy').replace('.STL', '.npy').replace('.obj', '.npy')
                        assert os.path.exists(vertex_path)
                        points_info = np.load(vertex_path)
                    else:
                        print(visual, link_name)
                        raise NotImplementedError

                    if self.make_contact_points:
                        idxs = np.arange(len(points_info))
                    else:
                        idxs = np.load(os.path.dirname(os.path.realpath(__file__)) + '/../assets/visible_point_indices/{}.npy'.format(link_name))

                    verts = link_pre_transform.transform_points(torch.FloatTensor(points_info[idxs, :3]))
                    verts *= torch.tensor(scale, dtype=torch.float)

                    vertex_normals = link_pre_transform.transform_normals(torch.FloatTensor(points_info[idxs, 3:6]))

                    temp = torch.ones(idxs.shape[0], 1)

                    meshes[link_name] = [
                        torch.cat((verts, temp), dim=-1).to(self.device),
                        torch.zeros([0]),  # no real meaning, just for placeholder
                        torch.cat((vertex_normals, temp), dim=-1).to(torch.float).to(self.device)
                    ]

        return meshes

    def get_hand_segment_indices(self):
        hand_segment_indices = {}
        hand_finger_indices = {}
        segment_start = 0  # torch.tensor(0, dtype=torch.long)
        finger_start = 0  # torch.tensor(0, dtype=torch.long)
        for link_name in self.order_keys:
            # end = torch.tensor(self.meshes[link_name][0].shape[0], dtype=torch.long) + segment_start
            end = self.meshes[link_name][0].shape[0] + segment_start
            hand_segment_indices[link_name] = [segment_start, end]
            if link_name in self.ordered_finger_endeffort:
                hand_finger_indices[link_name] = [finger_start, end]
                finger_start = end
            segment_start = end
        return hand_segment_indices, hand_finger_indices

    def get_complete_joint_angles(self, theta):
        bs = theta.shape[0]
        theta_complete = torch.zeros([bs, self.chain.n_joints], dtype=torch.float).to(self.device)
        theta_complete[:, self.activate_joints] = theta
        for item in self.mimic_joints_info:
            idx, mimic_idx, multiplier, offset, _, _ = item
            theta_complete[:, idx] = theta_complete[:, mimic_idx] * multiplier + offset
        return theta_complete

    def forward(self, theta):
        """
        Args:
            theta (Tensor (batch_size x 15)): The degrees of freedom of the Robot hand.
       """
        if len(self.mimic_joints) > 0:
            theta = self.get_complete_joint_angles(theta)

        ret = self.chain.forward_kinematics(theta)
        return ret

    def compute_abnormal_joint_loss(self, theta):
        loss_1 = torch.clamp(theta[:, 0], 0, 1) * 40
        loss_2 = torch.abs(0.4 - theta[:, 3]) * 40
        return loss_1 + loss_2

    def get_init_angle(self):
        init_angle = torch.tensor([0.4, 0.0,  # thumb
                                            0.15, # ring
                                            0.5,  # spread
                                            0.15,  # little
                                            0.0, 0.15,  # index
                                            0.0, 0.15,  # middle
                                            ], dtype=torch.float, device=self.device)
        return init_angle

    def get_hand_mesh(self, pose, ret):
        bs = pose.shape[0]
        meshes = []
        for key in self.order_keys:
            rotmat = ret[key].get_matrix()
            rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, rotmat))

            vertices = self.meshes[key][0]
            batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
            face = self.meshes[key][1]
            sub_meshes = [trimesh.Trimesh(vertices.cpu().numpy(), face) for vertices in batch_vertices]

            meshes.append(sub_meshes)

        hand_meshes = []
        for j in range(bs):
            hand = [meshes[i][j] for i in range(len(meshes))]
            hand_mesh = np.sum(hand)
            hand_meshes.append(hand_mesh)
        return hand_meshes

    def get_forward_hand_mesh(self, pose, theta):
        outputs = self.forward(theta)

        hand_meshes = self.get_hand_mesh(pose, outputs)

        return hand_meshes

    def get_forward_vertices(self, pose, theta):
        outputs = self.forward(theta)

        verts = []
        verts_normal = []

        # for key, item in self.meshes.items():
        for key in self.order_keys:
            rotmat = outputs[key].get_matrix()
            rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, rotmat))

            vertices = self.meshes[key][0]
            vertex_normals = self.meshes[key][2]
            batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
            verts.append(batch_vertices)

            if self.make_contact_points:
                if not os.path.exists('../assets/hand_composite_points'):
                    os.makedirs('../assets/hand_composite_points', exist_ok=True)
                np.save('../assets/hand_composite_points/{}.npy'.format(key),
                        batch_vertices.squeeze().cpu().numpy())
            rotmat[:, :3, 3] *= 0
            batch_vertex_normals = torch.matmul(rotmat, vertex_normals.transpose(0, 1)).transpose(1, 2)[..., :3]
            verts_normal.append(batch_vertex_normals)

        verts = torch.cat(verts, dim=1).contiguous()
        verts_normal = torch.cat(verts_normal, dim=1).contiguous()
        return verts, verts_normal


class SvhAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # vert_idx
        vert_idx = np.array([
            # thumb finger
            3429, 3510, 3804, 3817, 3818,
            1785, 2078,

            # index finger
            3916, 4113, 4314, 4261, 4321,
            2364,

            # middle finger
            4513, 4702, 4740, 4801, 4808,
            3029, 1637,

            # ring finger
            4863, 5199, 5291, 5266, 5223,
            2656, 2707,  # 2961

            # little finger
            5382, 5615, 5710, 5658, 5635,

            # plus side contact
            4136, 4079, 4152, 3976,
            4589, 4789, 4656, 4591,
            5075, 5064, 5103, 5012,
            5575, 5700,

        ])
        # vert_idx = np.load(os.path.join(BASE_DIR, 'anchor_idx.npy'))
        self.register_buffer("vert_idx", torch.from_numpy(vert_idx).long())

    def forward(self, vertices):
        """
        vertices: TENSOR[N_BATCH, 4040, 3]
        """
        anchor_pos = vertices[:, self.vert_idx, :]
        return anchor_pos

    def pick_points(self, vertices: np.ndarray, normals: np.ndarray, hand_segment_indices):
        import open3d as o3d
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        colors = np.ones((vertices.shape[0], 3))
        for segment in hand_segment_indices:
            start, end = hand_segment_indices[segment]
            random_color = np.random.choice(range(256), size=3) / 255.0
            colors[start:end] = random_color
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print(vis.get_picked_points())
        return vis.get_picked_points()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    show_mesh = False
    to_mano_frame = True
    hand = SvhHandLayer(show_mesh=show_mesh, to_mano_frame=to_mano_frame, device=device)

    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = np.zeros((1, hand.n_dofs), dtype=np.float32)
    # theta[0, 12:17] = np.array([-0.5, 0.5, 0, 0, 0])
    theta = torch.from_numpy(theta).to(device)
    # print(hand.joints_lower)
    # print(hand.joints_upper)
    theta = joint_angles_mu = torch.tensor([[0.4, 0.2,  # thumb
                                            0.2, # ring
                                            0.3,  # spread
                                            0.2,  # little
                                            0.0, 0.2,  # index
                                            0.0, 0.2,  # middle
                                            ]], dtype=torch.float, device=device)

    # mesh version
    if show_mesh:
        mesh = hand.get_forward_hand_mesh(pose, theta)[0]
        mesh.show()
    else:
        hand_segment_indices, hand_finger_indices = hand.get_hand_segment_indices()
        verts, normals = hand.get_forward_vertices(pose, theta)
        # print(verts.shape)
        pc = trimesh.PointCloud(verts.squeeze().cpu().numpy(), colors=(0, 255, 255))
        ray_visualize = trimesh.load_path(np.hstack((verts[0].detach().cpu().numpy(),
                                                     verts[0].detach().cpu().numpy() + normals[0].detach().cpu().numpy() * 0.01)).reshape(-1, 2, 3))

        mesh = trimesh.load(os.path.join(hand.BASE_DIR, '../assets/hand_to_mano_frame.obj'))
        scene = trimesh.Scene([mesh, pc, ray_visualize])
        scene.show()

        anchor_layer = SvhAnchor()
        anchor_layer.pick_points(verts.squeeze().cpu().numpy(), normals.squeeze().cpu().numpy(), hand_segment_indices)
        anchors = anchor_layer(verts).squeeze().cpu().numpy()
        pc_anchors = trimesh.PointCloud(anchors, colors=(0, 0, 255))
        ray_visualize = trimesh.load_path(np.hstack((verts[0].detach().cpu().numpy(),
                                                     verts[0].detach().cpu().numpy() + normals[
                                                         0].detach().cpu().numpy() * 0.01)).reshape(-1, 2, 3))

        scene = trimesh.Scene([mesh, pc, pc_anchors, ray_visualize])
        scene.show()

