import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from gaze_analysis import (unproject_points_2d, compute_angular_error, 
                          plot_angular_error_heatmap, plot_2d_scatter,
                          write_to_csv)

# Declare known variables from the instructions
K = np.array([[765.0, 0, 567.0], 
              [0, 765.0, 545.0], 
              [0, 0, 1]]) # known as the intrinsic parameters of the camera.

lens_distortion = np.array([[]])

# Session state management
if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_stage(stage):
    st.session_state.stage = stage

### Sidebar ###

# Logo
app_logo = Image.open("./Images/logo.png")
st.sidebar.image(app_logo)

# Files Upload
st.sidebar.subheader("Upload Gaze Data file (with columns timestamp, X, Y):")
gaze_datafile = st.sidebar.file_uploader(
    "Choose a gaze file from your local directory...",
    type=["csv"]
)

st.sidebar.subheader("Upload Target Data file (with columns timestamp, X, Y):")
target_datafile = st.sidebar.file_uploader(
    "Choose a target file from your local directory...",
    type=["csv"]
)

# Calculate angular error button
angular_error_button = st.sidebar.button("Compute angular error", 
                                         on_click= set_stage, 
                                         args = (1,))


### Main App ###

# Main title
st.title("Angular Error Calculation Demo")

# Manipulate datasets
columns = ["X [px]", "Y [px]"]
if gaze_datafile is not None:
    gaze_df = pd.read_csv(gaze_datafile, usecols = columns)
else:
    st.warning('Please upload a gaze data file', icon="⚠️")

if target_datafile is not None:
    target_df = pd.read_csv(target_datafile, usecols = columns)
else:
    st.warning('Please upload a target data file', icon="⚠️")

# Compute angular error
if st.session_state.stage > 0:
    gaze_3d = unproject_points_2d(gaze_df, 
                                  K, 
                                  dist_coefs = lens_distortion,
                                  normalize=False)

    target_3d = unproject_points_2d(target_df, 
                                    K, 
                                    dist_coefs = lens_distortion,
                                    normalize=False)

    angular_error_df = compute_angular_error(gaze_3d, target_3d)

    mean_angular_error = np.mean(angular_error_df)

    st.markdown("**On average, patient's gaze deviated from the target by "
          ":red[{:.2f}] degrees of visual angle.**".format(mean_angular_error))
    
    st.markdown("_To generate visualization plot, "
                "press the Generate Plot button below._")
    
    generate_plot_button = st.button("Generate plot", on_click= set_stage, args = (2,))
    
if st.session_state.stage > 1:
    gaze_3d = unproject_points_2d(gaze_df, 
                                  K, 
                                  dist_coefs = lens_distortion,
                                  normalize=False)

    target_3d = unproject_points_2d(target_df, 
                                    K, 
                                    dist_coefs = lens_distortion,
                                    normalize=False)

    dx = gaze_3d[:, 0] - target_3d[:, 0]
    dy = gaze_3d[:, 1] - target_3d[:, 1]

    heatmap, xedges, yedges = np.histogram2d(dx, dy, bins=60)

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

    
    st.pyplot(fig)

    # Reset app button
    st.markdown('_Reset the app to start from the beginning:_')
    st.button('Reset', on_click=set_stage, args=(0,))


    



