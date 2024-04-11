import argparse
import numpy as np
import os
from external.PanoAnnotator import utils
from external.PanoAnnotator import data
from PIL import Image
from functools import wraps
import matplotlib.pyplot as plt
from external.HorizonNet.dataset import cor_2_1d, find_occlusion
from scipy.spatial.distance import cdist


def _visualize_image(default_name, data_range=None, cmap=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, output_dir=None, output_prefix='', output_format='png', *args, **kwargs):
            image = func(self, *args, **kwargs)
            if output_dir:
                data_norm = image.max() if data_range is None else data_range
                image_norm = image / data_norm
                if cmap is not None:
                    image_save = (plt.get_cmap(cmap)(image_norm) * 255).astype(np.uint8)[..., :3]
                else:
                    image_save = (image_norm * 255).astype(np.uint8)

                if os.path.splitext(output_dir)[1] == '':
                    output_dir = os.path.join(output_dir, f"{output_prefix}{default_name}.{output_format}")
                output_folder = os.path.dirname(output_dir)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                Image.fromarray(image_save).save(output_dir)
            return image
        return wrapper
    return decorator


class Layout:
    default_size = (512, 1024)

    def __init__(self, scene):
        self.scene = scene
        self.distance_map = None
        self.plane_map = None

    @classmethod
    def from_json(cls, json_path):
        scene = data.Scene(None)
        scene.initEmptyScene()
        jdata = utils.loadLabelByJson(json_path, scene)
        layout = cls(scene)
        layout.jdata = jdata
        return layout

    @classmethod
    def from_layout_coords(cls, layout_coords, camera_height):
        # layout_coords: (N, 2, 2), in range [0, 1]
        layout_v = -(layout_coords[..., 1] - 0.5) * np.pi
        layout_dis = camera_height / np.tan(-layout_v[..., 1])
        layout_height = layout_dis * np.tan(layout_v[..., 0]) + camera_height
        layout_height = layout_height.mean()

        scene = data.Scene(None)
        scene.initEmptyScene()
        scene.label.setCameraHeight(camera_height)
        scene.label.setLayoutHeight(layout_height)

        gPoints = []
        layout_u = (layout_coords[:, 0, 0] - 0.5) * 2 * np.pi
        for point_u, point_dis in zip(layout_u, layout_dis):
            x = np.sin(point_u) * point_dis
            y = 0
            z = -np.cos(point_u) * point_dis
            xyz = (x, y, z)
            gPoint = data.GeoPoint(scene, None, xyz)
            gPoints.append(gPoint)

        scene.label.setLayoutPoints(gPoints)
        layout = cls(scene)
        layout.coords = layout_coords
        return layout

    def to_layout_coords(self):
        if hasattr(self, 'coords'):
            return self.coords

        # layout_coords: (N, 2, 2), in range [0, 1]
        us = np.array([pts['coords'][0] for pts in self.jdata['layoutPoints']['points']])
        cs = np.array([pts['xyz'] for pts in self.jdata['layoutPoints']['points']])
        cs = np.sqrt((cs**2)[:, [0, 2]].sum(1))

        vf = np.arctan2(-1.6, cs)
        vc = np.arctan2(-1.6 + self.jdata['layoutHeight'], cs)
        vf = (-vf / np.pi + 0.5)
        vc = (-vc / np.pi + 0.5)

        cor_x = np.repeat(us, 2)
        cor_y = np.stack([vc, vf], -1).reshape(-1)
        cor_xy = np.stack([cor_x, cor_y], -1)

        return cor_xy.reshape(-1, 2, 2)

    @classmethod
    def from_layout_pos(cls, layout_pos, camera_height, image_size):
        # layout_pos: (N, 2, 2), in range [0, image_size)
        image_size = np.array(image_size)
        layout_coords = layout_pos / image_size
        layout = cls.from_layout_coords(layout_coords, camera_height)
        layout.pos = layout_pos
        return layout

    def to_layout_pos(self, image_size):
        if hasattr(self, 'pos'):
            return self.pos

        image_size = np.array(image_size)
        layout_coords = self.to_layout_coords()
        return layout_coords * image_size

    def to_horizonnet(self, image_size):
        cor = self.to_layout_pos(image_size).reshape(-1, 2)

        # Detect occlusion
        occlusion = find_occlusion(cor[::2].copy()).repeat(2)

        # Prepare 1d ceiling-wall/floor-wall boundary
        bon = cor_2_1d(cor, image_size[1], image_size[0])

        # Prepare 1d wall-wall probability
        corx = cor[~occlusion, 0]
        dist_o = cdist(corx.reshape(-1, 1), np.arange(image_size[0]).reshape(-1, 1), 'minkowski', p=1)
        dist_r = cdist(corx.reshape(-1, 1), np.arange(image_size[0]).reshape(-1, 1) + image_size[0], 'minkowski', p=1)
        dist_l = cdist(corx.reshape(-1, 1), np.arange(image_size[0]).reshape(-1, 1) - image_size[0], 'minkowski', p=1)
        dist = np.min([dist_o, dist_r, dist_l], 0)
        nearest_dist = dist.min(0)
        y_cor = (0.96 ** nearest_dist).reshape(1, -1)

        return {'bon': bon.astype(np.float32), 'cor': y_cor.astype(np.float32)}

    def render_layout(self, layout_types=None, output_dir=None, output_prefix='', size=default_size):
        if layout_types is None:
            layout_types = ['wireframe', 'edge_map', 'orientation_map', 'normal_map', 'distance_map', 'object2d_map']

        results = {}
        for layout_type in layout_types:
            results[layout_type] = getattr(self, f'render_{layout_type}')(output_dir=output_dir, output_prefix=output_prefix, size=size)
        return results

    @_visualize_image(default_name='wireframe', data_range=255)
    def render_wireframe(self, background=None, size=default_size, color=None):
        if background is None:
            edgeMap = np.zeros(size, dtype=np.uint8)
            color = color or (255,)
        else:
            size = background.shape[:2]
            edgeMap = background.copy()
            color = color or [0] * background.shape[2]

        sizeT = (size[1],size[0])
        for wall in self.scene.label.getLayoutWalls():
            # if wall.planeEquation[3] > 0:
            #     continue
            for edge in wall.edges:
                for i in range(len(edge.coords)-1):
                    isCross, l, r = utils.pointsCrossPano(edge.sample[i],
                                                        edge.sample[i+1])
                    if not isCross:
                        pos1 = utils.coords2pos(edge.coords[i], sizeT)
                        pos2 = utils.coords2pos(edge.coords[i+1], sizeT)
                        utils.imageDrawLine(edgeMap, pos1, pos2, color)
                    else:
                        lpos = utils.coords2pos(utils.xyz2coords(l), sizeT)
                        rpos = utils.coords2pos(utils.xyz2coords(r), sizeT)
                        ch = int((lpos[1] + rpos[1])/2)
                        utils.imageDrawLine(edgeMap, lpos, (0,ch), color)
                        utils.imageDrawLine(edgeMap, rpos, (sizeT[0],ch), color)

        return edgeMap

    @_visualize_image(default_name='edge_map', data_range=1.)
    def render_edge_map(self, size=default_size):
        return utils.genLayoutEdgeMap(self.scene, (*size, 3))

    def render_plane_map(self, size=default_size):
        if self.plane_map is None:
            self.render_distance_map(size=size)
        return self.plane_map

    @_visualize_image(default_name='orientation_map', data_range=1.)
    def render_orientation_map(self, size=default_size):
        return utils.genLayoutOMap(self.scene, (*size, 3), self.render_plane_map(size))

    @_visualize_image(default_name='normal_map', data_range=1.)
    def render_normal_map(self, size=default_size):
        return utils.genLayoutNormalMap(self.scene, (*size, 3), self.render_plane_map(size))

    @_visualize_image(default_name='distance_map_vis', cmap='jet')
    def render_distance_map(self, size=default_size):
        if self.plane_map is None:
            self.distance_map, self.plane_map = utils.genLayoutDepthMap(self.scene, size, True)
        return self.distance_map

    @_visualize_image(default_name='object2d_map', data_range=1.)
    def render_object2d_map(self, size=default_size):
        return utils.genLayoutObj2dMap(self.scene, (*size, 3))


def parse_args():
    parser = argparse.ArgumentParser(description="Test Layout Class with Matterport3D Dataset")
    parser.add_argument("--panorama_dir", type=str, required=True, help="Path to panorama directory")
    parser.add_argument("--mp3d_anno_dir", type=str, default='data/Matterport3DLayoutAnnotation/label_data',
                        help="Path to Matterport3DLayoutAnnotation directory")
    return parser.parse_args()


def main():
    args = parse_args()

    scene_id, _, file_name = args.panorama_dir.split('/')[-3:]
    view_id = os.path.splitext(file_name)[0]
    json_dir = os.path.join(args.mp3d_anno_dir, f"{scene_id}_{view_id}_label.json")
    layout = Layout.from_json(json_dir)
    layout_pos = layout.to_layout_pos((2048, 1024))
    layout = Layout.from_layout_pos(layout_pos, 1.6, (2048, 1024))
    image = np.array(Image.open(args.panorama_dir).convert('RGB'))

    layout.render_wireframe(output_dir='debug/wireframe_overlay.png', background=image)
    layout.render_layout(output_dir='debug')


if __name__ == '__main__':
    main()
