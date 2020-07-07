import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.preprocessing import normalize
from progress.bar import Bar
from time import sleep
from mpl_toolkits import mplot3d
from open3d import *



DPI = 96
DATASET = "dataset"
DATASET_LEFT = DATASET + "/left/"
DATASET_RIGHT = DATASET + "/right/"
DATASET_DISPARITIES = DATASET + "/disparities/"
DATASET_COMBINED = DATASET + "/combined/"
DATASET_POINT_CLOUDS = DATASET + "/point_clouds/"
data_folder_calib = "/calib/"
calib_fname = 'um_000000' + '.txt'


def process_frame(left, right, name):
    kernel_size = 3
    smooth_left = cv2.GaussianBlur(left, (kernel_size, kernel_size), 1.5)
    smooth_right = cv2.GaussianBlur(right, (kernel_size, kernel_size), 1.5)

    window_size = 9
    left_matcher = cv2.StereoSGBM_create(
        numDisparities=96,
        blockSize=3,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=16,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)

    disparity_left = np.int16(left_matcher.compute(smooth_left, smooth_right))
    disparity_right = np.int16(right_matcher.compute(smooth_right, smooth_left))

    wls_image = wls_filter.filter(disparity_left, smooth_left, None, disparity_right)

    wls_image = cv2.normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    wls_image = np.uint8(wls_image)

    create_point_cloud(left, right, name, disparity_left)

    fig = plt.figure(figsize=(wls_image.shape[1] / DPI, wls_image.shape[0] / DPI), dpi=DPI, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(wls_image, cmap='jet')
    plt.savefig(DATASET_DISPARITIES + name)
    plt.close()
    create_combined_output(left, right, name)


def create_point_cloud(left, right, name, img):
    # Reading calibration
    matrix_type_1 = 'P2'
    matrix_type_2 = 'P3'

    calib_file = DATASET + data_folder_calib + calib_fname
    with open(calib_file, 'r') as f:
        fin = f.readlines()
        for line in fin:
            if line[:2] == matrix_type_1:
                calib_matrix_1 = np.array(line[4:].strip().split(" ")).astype('float32').reshape(3, -1)
            elif line[:2] == matrix_type_2:
                calib_matrix_2 = np.array(line[4:].strip().split(" ")).astype('float32').reshape(3, -1)
    # Calculate depth-to-disparity
    cam1 = calib_matrix_1[:, :3]  # left image - P2
    cam2 = calib_matrix_2[:, :3]  # right image - P3

    Tmat = np.array([0.54, 0., 0.])

    rev_proj_matrix = np.zeros((4, 4))

    cv2.stereoRectify(cameraMatrix1=cam1, cameraMatrix2=cam2,
                      distCoeffs1=0, distCoeffs2=0,
                      imageSize=left.shape[:2],
                      R=np.identity(3), T=Tmat,
                      R1=None, R2=None,
                      P1=None, P2=None, Q=rev_proj_matrix)
    points = cv2.reprojectImageTo3D(img, rev_proj_matrix)

    # reflect on x axis
    reflect_matrix = np.identity(3)
    reflect_matrix[0] *= -1
    points = np.matmul(points, reflect_matrix)

    # extract colors from image
    colors = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

    # filter by min disparity
    mask = img > img.min()
    out_points = points[mask]
    out_colors = colors[mask]

    # filter by dimension
    idx = np.fabs(out_points[:, 0]) < 4.5
    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]

    name = name.split('.')
    print(name)
    write_ply(DATASET_POINT_CLOUDS + name[0] + '.ply', out_points, out_colors)
    # print('%s saved' % 'out.ply')
    # reflected_pts = np.matmul(out_points, reflect_matrix)
    # projected_img, _ = cv2.projectPoints(reflected_pts, np.identity(3), np.array([0., 0., 0.]),
    #                                      cam2[:3, :3], np.array([0., 0., 0., 0.]))
    # projected_img = projected_img.reshape(-1, 2)
    # blank_img = np.zeros(left.shape, 'uint8')
    # img_colors = right[mask][idx].reshape(-1, 3)
    #
    # for i, pt in enumerate(projected_img):
    #     pt_x = int(pt[0])
    #     pt_y = int(pt[1])
    #     if pt_x > 0 and pt_y > 0:
    #         # use the BGR format to match the original image type
    #         col = (int(img_colors[i, 2]), int(img_colors[i, 1]), int(img_colors[i, 0]))
    #         cv2.circle(blank_img, (pt_x, pt_y), 1, col)
    # plt.imshow(cv2.cvtColor(blank_img, cv2.COLOR_RGB2BGR))

    # cloud = read_point_cloud(DATASET_POINT_CLOUDS + '000000_00.ply')  # Read the point cloud
    # draw_geometries([cloud])  # Visualize the point cloud


def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.copy()
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def create_combined_output(left, right, name):
    combined = np.concatenate((left, right, cv2.imread(DATASET_DISPARITIES + name)), axis=0)
    cv2.imwrite(DATASET_COMBINED + name, combined)


def process_dataset():
    left_images = [f for f in os.listdir(DATASET_LEFT) if not f.startswith('.')]
    right_images = [f for f in os.listdir(DATASET_RIGHT) if not f.startswith('.')]
    assert (len(left_images) == len(right_images))
    left_images.sort()
    right_images.sort()

    with Bar('Processing', max=len(left_images)) as bar:
        for i in range(len(left_images)):
            left_image_path = DATASET_LEFT + left_images[i]
            right_image_path = DATASET_RIGHT + right_images[i]
            left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
            right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
            process_frame(left_image, right_image, left_images[i])
            bar.next()
    bar.finish()
    print("Finished")

    # cloud = io.read_point_cloud(DATASET_POINT_CLOUDS + '001472.ply')  # Read the point cloud
    # io.draw_geometries([cloud], width=1920, height=1080, left=0, top=0)  # Visualize the point cloud



if __name__ == "__main__":
    process_dataset()
    # cloud = read_point_cloud(DATASET_POINT_CLOUDS + '001472.ply')  # Read the point cloud
    # draw_geometries([cloud], width=1920, height=1080, left=50, top=50)  # Visualize the point cloud
