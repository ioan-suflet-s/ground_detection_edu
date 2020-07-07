import numpy as np
import open3d as o3d
from open3d import *
import pcl
import os
from progress.bar import Bar

DATASET = "dataset"
DATASET_POINT_CLOUDS = DATASET + "/point_clouds/"


def display_inlier_outlier(cloud, ind):
    inlier_cloud = geometry.PointCloud.select_down_sample(cloud, ind, invert=True)
    outlier_cloud = geometry.PointCloud.select_down_sample(cloud, ind)

    # print("Showi¬g outliers (red) and inliers (gray): ")
    outlier_cloud = geometry.PointCloud.paint_uniform_color(outlier_cloud, [1, 0, 0])
    # inlier_cloud = geometry.PointCloud.paint_uniform_color(inlier_cloud, [0.8, 0.8, 0.8])
    visualization.draw_geometries([inlier_cloud, outlier_cloud])


def down_sample(cloud):
    print("Downsample the point cloud with a voxel of 0.01")
    rez = geometry.PointCloud.voxel_down_sample(cloud, voxel_size=0.001)
    visualization.draw_geometries([rez])

    # print("Every 5th points are selected")
    # uni_down_pcd = uniform_down_sample(pcd, every_k_points=5)
    # draw_geometries([uni_down_pcd])
    return rez


def remove_outliers(cloud):
    # print("Statistical oulier removal")
    # cl, ind = geometry.PointCloud.statistical_outlier_removal(voxel_down_pcd,
    #                                                           nb_neighbors=20, std_ratio=2.0)

    print("Radius oulier removal")
    rez, ind = geometry.PointCloud.remove_radius_outlier(cloud, nb_points=4, radius=0.08)
    #display_inlier_outlier(rez,ind)
    return rez


def process_dataset():
    point_clouds = [f for f in os.listdir(DATASET_POINT_CLOUDS) if not f.startswith('.')]
    point_clouds.sort()

    with Bar('Processing', max=len(point_clouds)) as bar:
        for i in range(len(point_clouds)):
            point_clouds_path = DATASET_POINT_CLOUDS + point_clouds[i]
            print("\nLoad" + point_clouds_path)
            pcd = io.read_point_cloud(point_clouds_path)
            voxel_down_pcd = down_sample(pcd)
            # voxel_down_pcd = pcd
            pcd_no_outliers = remove_outliers(voxel_down_pcd)
            print("Plane Segment")
            plane_model, inliers = geometry.PointCloud.segment_plane(pcd_no_outliers, distance_threshold=0.01,
                                                                     ransac_n=3,
                                                                     num_iterations=2000)
            [a, b, c, d] = plane_model
            print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            display_inlier_outlier(pcd_no_outliers, inliers)

            bar.next()
    bar.finish()
    print("Finished")


if __name__ == "__main__":
    process_dataset()
