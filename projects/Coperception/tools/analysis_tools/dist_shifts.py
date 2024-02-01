import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from projects.Coperception.tools.visualize_data import load_points

pkl_path = 'data/openv2v/openv2v_infos_test_x.pkl'
# pkl_path = 'data/v2v4real/v2v4real_infos_test_x.pkl'
sns.set(style='whitegrid')


def split_points_numpy(points: np.ndarray, use_dim=3):
    points = points.astype(np.float32)
    cav_ids = np.unique(points[:, -1].astype(np.int32))
    print(cav_ids)
    points_list = []
    for idx, label in enumerate(cav_ids):
        points_list.append(points[points[:, -1] == label])
    points_list = [points[:, :use_dim] for points in points_list]
    return points_list


def draw_dist_shifts(sample):
    # print(sample)
    points = load_points(
        sample['lidar_points'], load_dim=5, use_dim=5)['points']
    point_clouds = split_points_numpy(points)

    # Create a figure and axis
    plt.figure(figsize=(14, 6))
    plt.rcParams['font.size'] = 12  # Set font size

    # Plot each point cloud with a different color and update the legend
    legend_labels = ['Vehicle in Front', 'Ego Vehicle', 'Vehicle Behind']
    for i, cloud in enumerate(point_clouds):
        cloud_distances = np.linalg.norm(cloud, axis=1)
        cloud_signed_distances = cloud_distances * np.sign(
            cloud[:, 0])  # Use X-coordinate for sign

        # densities = np.ones_like(cloud_signed_distances) * len(cloud)  # Density proportional to point count

        # Create a histogram for the current point cloud
        num_bins = 250
        hist, edges = np.histogram(
            cloud_signed_distances, bins=num_bins, density=True)

        # Plot the histogram
        plt.bar(
            edges[:-1],
            hist,
            width=np.diff(edges),
            align='edge',
            alpha=0.5,
            label=legend_labels[i])

    # Set x-axis limits
    plt.xlim([-100, 100])

    # Add labels, legend, and grid
    plt.xlabel('Distance to Ego Vehicle (m)', fontweight='bold', fontsize=22)
    plt.ylabel('Density', fontweight='bold', fontsize=22)
    plt.xticks(range(-80, 81, 20), fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)

    plt.savefig('dist_shifts.png', dpi=600, bbox_inches='tight')
    # Show the plot
    plt.show()


if __name__ == '__main__':
    with open(pkl_path, 'rb') as f:
        data_list = pickle.load(f)['data_list']
    while True:
        # i = random.randint(0, len(data_list) - 1)
        i = 900
        print(f'visualizing sample {i}')
        sample = data_list[i]
        draw_dist_shifts(sample)
        break
