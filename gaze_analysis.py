import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2

def unproject_points_2d(
        points, camera_matrix, dist_coefs=None, normalize=False
        ):
    """
    # source: https://github.com/pupil-labs/pupil-tutorials/blob/master/11_undistortion_and_unprojection.ipynb
    
    Transforms points from the distorted or undistorted image space to the 
    undistorted camera space.

    Input
    -----
    points : array-like, shape (N, 2)
        2D points in either distorted image space (when dist_coefs are 
        provided) or undistorted image space.
        
    camera_matrix : array-like, shape (3, 3)
        The camera matrix containing the intrinsic parameters of the camera.
        
    dist_coefs : array-like, optional, shape (1, 5) or (5,)
        The distortion coefficients (k1, k2, p1, p2, k3) of the camera.
        If None (default), it is assumed that the input points are in the 
        undistorted image space.
        
    normalize : bool, optional, default: False
        Determines whether the output points lie on a plane (False) 
        or a sphere (True).

    Output
    ------
    3d_points : ndarray, shape (N, 3)
        The 3D points in the undistorted camera space.
    """
    points = np.array(points, dtype="float64")
    camera_matrix = np.array(camera_matrix)
    
    if dist_coefs is None:
        dist_coefs = np.array([[]])
    else:
        dist_coefs = np.array(dist_coefs)

    points = cv2.undistortPoints(points, camera_matrix, dist_coefs)
    points_3d = cv2.convertPointsToHomogeneous(points).reshape(-1, 3)

    if normalize:
        points_3d /= np.linalg.norm(points_3d, axis=1)[:, np.newaxis]

    return points_3d


def compute_angular_error(array1_3d, array2_3d):
    """
    Calculates the angular error between two arrays in 3D space.

    Input
    -----
    array1_3d : array-like, shape (N, 3)
        points in 3D, where N is the number of points.
    
    array2_3d : array-like, shape (N, 3)
        points in 3D, where N is the number of points.

    Output
    ------
    angular_error_in_degrees : ndarray, shape (N,)
        The angular error between the array1 points and the array2 
        points in degrees.

    Notes
    -----
    This function assumes no lens distortion.
    """
    cos_angle = ( np.sum(array1_3d * array2_3d, axis=1) / 
                 (np.linalg.norm(array1_3d, axis=1) * 
                  np.linalg.norm(array2_3d, axis=1))
                 )
    angle_rad = np.arccos(cos_angle)
    angular_error_in_degrees = np.degrees(angle_rad)
    
    return angular_error_in_degrees


def plot_angular_error_heatmap(
        array1_3d, array2_3d, bins=60, output_filename = None
        ):
    """
    Plots a heatmap of the angular error distribution between gaze and 
    target points. The wider the distribution around the center the larger 
    the error. 

    Input
    -----
    array1_3d : array-like, shape (N, 3)
        3D gaze points, where N is the number of points.
    
    array2_3d : array-like, shape (N, 3)
        3D target points, where N is the number of points.

    bins : int, optional, default: 60
        The number of bins along each axis for the 2D histogram.
    
    output_filename : str, optional, default: None
        The filename for the saved heatmap plot image.

    Output
    ------
    Image of the plot.
    """
    dx = array1_3d[:, 0] - array2_3d[:, 0]
    dy = array1_3d[:, 1] - array2_3d[:, 1]

    heatmap, xedges, yedges = np.histogram2d(dx, dy, bins=bins)

    # Plot the heatmap
    fig, ax = plt.subplots()
    pos = ax.imshow(heatmap.T, 
                    origin="lower", 
                    cmap="viridis", 
                    extent=[xedges[0], 
                            xedges[-1], 
                            yedges[0], 
                            yedges[-1]])
    plt.colorbar(pos, ax=ax)
    ax.set_title("Angular Error Distribution", size = 12)
    ax.set_xlabel("Normalized error in X", size = 9)
    ax.set_ylabel("Normalized error in Y", size = 9)

    heatmap_image = plt.savefig(output_filename)
    plt.show()

    return heatmap_image

def plot_2d_scatter(
        arr1, arr2, screen_res, markersize=10, 
        title = None, xlabel = None, ylabel = None, 
        xlegend = None, ylegend = None
        ):
    """
    Plots together the X, Y coordinates of two arrays as scatterplots.
    
    Input
    -----
    arr1 : array-like, shape (N, 2)
        First set of 2D points, where N is the number of points.

    arr2 : array-like, shape (M, 2)
        Second set of 2D points, where M is the number of points.
        
    screen_res: tuple in the form of (width, height) 
        Screen resolution in pixel units.

    markersize : float, optional, default: 10
        Size of the scatterplot markers.

    title : str, optional, default: None
        Title for the plot. If not provided, no title will be displayed.

    xlabel : str, optional, default: None
        Label for the X-axis. If not provided, no label will be displayed.

    ylabel : str, optional, default: None
        Label for the Y-axis. If not provided, no label will be displayed.
    
    xlegend : str, optional, default: None
        Label for the legend in the X-axis. If not provided, 
        default will be displayed.
        
    ylegend : str, optional, default: None
        Label for the legend in the Y-axis. If not provided, 
        default will be displayed.

    Output
    ------
    None
    """
    fig, ax = plt.subplots(figsize = (9,9))
    
    ax.scatter(arr1[:, 0], 
               arr1[:, 1], 
               s = markersize, 
               color = "blue", 
               label= xlegend)
    ax.scatter(arr2[:, 0], 
               arr2[:, 1], 
               s = markersize, 
               color = "red", 
               label= ylegend)
    
    if title is not None:
        ax.set_title(title)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    ax.set_xlim(0, screen_res[0])
    ax.set_ylim(0, screen_res[1])
    plt.gca().invert_yaxis() # coordinates origin is top-left
    ax.legend()
    plt.show()
    
    return fig, ax



def write_to_csv(
        item_to_write, header = None, output_filename = "default.csv"
        ):
    """
    Writes an item to a CSV file with a header.

    Input
    -----
    item_to_write : object
        The item you want to write.
    
    header : str, optional, default: None
        The header of the CSV file.

    output_filename : str, optional, default: "default.csv"
        The filename for the output CSV file.

    Returns
    -------
    None
    """
    first_row_title = [header]

    with open(output_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(first_row_title)
        csv_writer.writerow([item_to_write])
    
    return output_filename





