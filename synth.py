import numpy as np
import poselib
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from theory.lo_verification import skew
from utils.geometry import rotation_angle, angle


def generate_points(num_pts, f, distance, depth, width=640, height=480):
    zs = (1 + distance) * f + depth * np.random.rand(num_pts) * f
    xs = (np.random.rand(num_pts) - 0.5) * width * (1 + distance)
    ys = (np.random.rand(num_pts) - 0.5) * height * (1 + distance)
    return np.column_stack([xs, ys, zs, np.ones_like(xs)])

def get_projection(P, X):
    x = P @ X.T
    x = x[:2, :] / x[2, np.newaxis, :]
    return x.T

def visible_in_view(x, width=640, height=480):
    visible = np.logical_and(np.abs(x[:, 0]) <= width / 2, np.abs(x[:, 1]) <= height / 2)
    return visible


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_scene(points, R, t, f, width=640, height=480, color_1='black', color_2='red', name=""):
    c_x_1 = np.array([0.5 * width, 0.5 * width, -0.5 * width, -0.5 * width, 0])
    c_y_1 = np.array([0.5 * height, -0.5 * height, -0.5 * height, 0.5 * height, 0])
    c_z_1 = np.array([f, f, f, f, 0])
    c_z_2 = np.array([f, f, f, f, 0])

    camera2_X = np.row_stack([c_x_1, c_y_1, c_z_2, np.ones_like(c_x_1)])
    c_x_2, c_y_2, c_z_2 = np.column_stack([R.T, -R.T @ t]) @ camera2_X

    # fig = plt.figure()

    ax = plt.axes(projection="3d")
    ax.set_box_aspect([1.0, 1., 1.0])

    ax.plot3D(c_x_1, c_y_1, c_z_1, color_1)
    ax.plot3D(c_x_2, c_y_2, c_z_2, color_2)

    ax.plot3D([c_x_1[0], c_x_1[3]], [c_y_1[0], c_y_1[3]], [c_z_1[0], c_z_1[3]], color_1)
    ax.plot3D([c_x_2[0], c_x_2[3]], [c_y_2[0], c_y_2[3]], [c_z_2[0], c_z_2[3]], color_2)

    for i in range(4):
        ax.plot3D([c_x_1[i], c_x_1[-1]], [c_y_1[i], c_y_1[-1]], [c_z_1[i], c_z_1[-1]], color_1)
        ax.plot3D([c_x_2[i], c_x_2[-1]], [c_y_2[i], c_y_2[-1]], [c_z_2[i], c_z_2[-1]], color_2)

    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c='blue')

    set_axes_equal(ax)

    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(name)


def get_scene(f, R1, t1, R2, t2, num_pts, X=None, min_distance=2, depth=1, width=640, height=480, sigma_p=0.0, plot=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    K = np.diag([f, f, 1])


    P1 = K @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = K @ np.column_stack([R1, t1])
    P3 = K @ np.column_stack([R2, t2])

    if X is None:
        X = generate_points(3 * num_pts, f, min_distance, depth, width=width, height=height)
    x1 = get_projection(P1, X)
    x2 = get_projection(P2, X)
    x3 = get_projection(P3, X)

    # visible = visible_in_view(x2, width=width, height=height)

    # x1, x2, X = x1[visible][:num_pts], x2[visible][:num_pts], X[visible]

    # run(f1, f2, x1, x2, scale=scale, name=name)
    # if plot is not None:
    #     plot_scene(X, R1, t1, f1, f2, name=plot)
    #     # plt.savefig(f'{np.round(t)}.png')
    #     plt.show()

    return x1, x2, x3, X


def run_synth():
    f = 600
    R12 = Rotation.from_euler('xyz', (0, 60, 0), degrees=True).as_matrix()
    R13 = Rotation.from_euler('xyz', (0, -30, 0), degrees=True).as_matrix()
    c1 = np.array([2 * f, 0, f])
    c2 = np.array([0, f, 0.5 * f])
    # R = Rotation.from_euler('xyz', (theta, 30, 0), degrees=True).as_matrix()
    # c = np.array([f1, y, 0])
    t12 = -R12 @ c1
    t13 = -R13 @ c2

    x1, x2, x3, X = get_scene(f, R12, t12, R13, t13, 100)

    sigma = 0.5

    # x1 += sigma * np.random.randn(*(x1.shape))
    # x2 += sigma * np.random.randn(*(x1.shape))
    # x3 += sigma * np.random.randn(*(x1.shape))

    idxs1 = np.random.permutation(np.arange(30))
    x1[:30] = x1[idxs1]
    idxs2 = np.random.permutation(np.arange(30, 60))
    x2[30:60] = x2[idxs2]


    T12 = np.diag([0, 0, 0, 1.0])
    T12[:3, :3] = R12
    T12[:3, 3] = t12
    T13 = np.diag([0, 0, 0, 1.0])
    T13[:3, :3] = R13
    T13[:3, 3] = t13

    T23 = T13 @ np.linalg.inv(T12)
    R23 = T23[:3, :3]
    t23 = T23[:3, 3]

    camera_dict =  {'model': 'SIMPLE_PINHOLE', 'width': 640, 'height': 480, 'params': [f, 0, 0]}

    # print(out)
    # print('**********')

    # ransac_dict = {'max_epipolar_error': 2.0, 'progressive_sampling': False,
    #                'min_iterations': 100, 'max_iterations': 10000, 'lo_iterations': 25,
    #                'inner_refine': False, 'threeview_check': True, 'sample_sz': 6,
    #                'delta': 0.025}
    #
    # pose, out6 = poselib.estimate_three_view_shared_focal_relative_pose(x1, x2, x3, np.array([0.0, 0.0]), ransac_dict, {'verbose': False})
    # print("Rot errs 6p")
    # print(pose.camera.focal())
    # print(rotation_angle(pose.poses.pose12.R.T @ R12))
    # print(rotation_angle(pose.poses.pose13.R.T @ R13))
    # print(out6['num_inliers'])
    #
    # ransac_dict['sample_sz'] = 5
    # pose, out5 = poselib.estimate_three_view_shared_focal_relative_pose(x1, x2, x3, np.array([0, 0]), ransac_dict, {'verbose': False})
    # print("Rot errs 5p")
    # print(pose.camera.focal())
    # print(rotation_angle(pose.poses.pose12.R.T @ R12))
    # print(rotation_angle(pose.poses.pose13.R.T @ R13))
    #
    # ransac_dict['sample_sz'] = 4
    # pose, out4 = poselib.estimate_three_view_shared_focal_relative_pose(x1, x2, x3, np.array([0, 0]), ransac_dict, {'verbose': False})
    # print("Rot errs 4p")
    # print(pose.camera.focal())
    # print(rotation_angle(pose.poses.pose12.R.T @ R12))
    # print(rotation_angle(pose.poses.pose13.R.T @ R13))
    #
    # return out6['iterations'], out5['iterations'], out4['iterations']

    ransac_dict = {'max_epipolar_error': 2.0, 'progressive_sampling': False,
                   'min_iterations': 100, 'max_iterations': 10000, 'lo_iterations': 25,
                   'inner_refine': False, 'threeview_check': True, 'sample_sz': 5,
                   'delta': 0.025, 'use_hc': False}

    # pose, out5 = poselib.estimate_three_view_relative_pose(x1, x2, x3, camera_dict, camera_dict, camera_dict, ransac_dict, {'verbose': False})
    # print("Rot errs 5p")
    # print(rotation_angle(pose.pose12.R.T @ R12))
    # print(rotation_angle(pose.pose13.R.T @ R13))
    #
    # ransac_dict['sample_sz'] = 4
    #
    # pose, out4 = poselib.estimate_three_view_relative_pose(x1, x2, x3, camera_dict, camera_dict, camera_dict, ransac_dict, {'verbose': False})
    # print("Rot errs 4p")
    # print(rotation_angle(pose.pose12.R.T @ R12))
    # print(rotation_angle(pose.pose13.R.T @ R13))

    # ransac_dict['use_net'] = False
    # ransac_dict['use_init'] = True
    ransac_dict['sample_sz'] = 4
    ransac_dict['gt_E'] = skew(t12) @ R12
    print(ransac_dict['gt_E'])
    pose, outR = poselib.estimate_three_view_relative_pose(x1, x2, x3, camera_dict, camera_dict, camera_dict, ransac_dict, {'verbose': False})
    # if (angle(pose.pose23().t, t23) > 1):
    print("Rot errs L")
    print(rotation_angle(pose.pose12.R.T @ R12))
    print(rotation_angle(pose.pose13.R.T @ R13))
    print(rotation_angle(pose.pose23().R.T @ R23))

    print(angle(pose.pose12.t, t12))
    print(angle(pose.pose13.t, t13))
    print(angle(pose.pose23().t, t23))
    print(outR['num_inliers'])

    # return out5['iterations'], out4['iterations'], outR['iterations']
    return


if __name__ == '__main__':
    iters = [run_synth() for _ in range(100)]
    iters6 = [x[0] for x in iters]
    iters5 = [x[1] for x in iters]
    iters4 = [x[2] for x in iters]
    print(f"Mean iters6: {np.mean(iters6)} - Median iters6: {np.nanmedian(iters6)}")
    print(f"Mean iters5: {np.mean(iters5)} - Median iters5: {np.nanmedian(iters5)}")
    print(f"Mean iters4: {np.mean(iters4)} - Median iters4: {np.nanmedian(iters4)}")