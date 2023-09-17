import os
import numpy as np

def compute_values(point_cloud, classification_column, height_column, point_cloud_area, pulse_area, scan_angle_column):
    classifications = point_cloud[:, classification_column]
    heights = point_cloud[:, height_column]

    # Compute the gap probability (P_gap)
    P_gap = np.count_nonzero(classifications == 2) / len(classifications)

    # Compute the effective extinction coefficient (k_eff)
    k_eff = np.abs(np.log(len(point_cloud) / (point_cloud_area / pulse_area))) / (2 * np.cos(np.deg2rad(point_cloud[:, scan_angle_column])))

    # Compute the derivative of the leaf area profile (dρ(z)/dz)
    d_rho_dz = np.gradient(heights)

    # Compute the vertical leaf area density profile (ρ(z))
    rho_z = d_rho_dz / k_eff

    # Compute the Leaf Area Index (LAI)
    T = np.count_nonzero(classifications != 2)
    T0 = np.count_nonzero(classifications == 2)
    omega = 1 - P_gap
    LAI = np.log(T / T0) / omega

    # Compute the clumping index
    clumping_index = np.var(heights) / (np.mean(heights) ** 2)

    return P_gap, k_eff, rho_z, LAI, clumping_index

def compute_lpi_pnb(point_cloud, classification_column):
    # Compute the number of ground and vegetation returns
    N_g = np.count_nonzero(point_cloud[:, classification_column] == 2)
    N_v = np.count_nonzero(point_cloud[:, classification_column] != 2)

    # Compute the Laser Penetration Index (LPI) using the Point-Number-Based (PNB) method
    LPI_pnb = N_g / (N_g + N_v)

    return LPI_pnb

def process_data(directory, scan_angle_column, classification_column, height_column):
    # Define the lidar sensor parameters
    sensor_height = 1000  # Height of the lidar sensor in meters
    footprint_diameter = 0.2  # Diameter of the lidar sensor's footprint in meters
    pulse_area = np.pi * (footprint_diameter / 2) ** 2
    point_cloud_area = 0

    file_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            file_paths.append(file_path)

    results = []
    for file_path in file_paths:
        # Load the point cloud data from the .txt file
        point_cloud = np.loadtxt(file_path, skiprows=2)

        # Compute the total point cloud area (if not already computed)
        if point_cloud_area == 0:
            min_x, max_x = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
            min_y, max_y = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
            point_cloud_area = (max_x - min_x) * (max_y - min_y)

        # Compute the values for each file separately
        P_gap, k_eff, rho_z, LAI, clumping_index = compute_values(point_cloud, classification_column, height_column, point_cloud_area, pulse_area, scan_angle_column)

        # Compute the Laser Penetration Index (LPI) using the Point-Number-Based (PNB) method
        LPI_pnb = compute_lpi_pnb(point_cloud, classification_column)

        # Store the computed values for the file
        result = {
            "File": file_path,
            "Gap Probability (P_gap)": P_gap,
            "Effective Extinction Coefficient (k_eff)": np.median(k_eff),
            "Clumping Index": clumping_index,
            "Laser Penetration Index (PNB)": LPI_pnb,
            "Leaf Area Index (LAI)": LAI
        }
        results.append(result)

    return results
