# /* Mehmet Mert Özgün S009580 Department of Computer Science */
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def draw_flow(img, flow, step=8):
    """
    Taken from: https://github.com/opencv/opencv/blob/master/samples/python/opt_flow.py
    :param img: Frame to draw the flow
    :param flow: Flow vectors
    :param step: Number of pixels each vector represents
    :return: visualisation
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_, _) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def dense_lk(cap, bin_count):
    """
    Applies lucas kanade optical flow to all pixels and creates a temporal mean histogram
    :param cap: Video
    :param bin_count: Bin count
    :return: Temporal mean histogram
    """
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Create empty matrices to fill later
    hist_magnitudes = np.zeros([bin_count, 1])
    bounds = np.zeros([bin_count, 2])

    # Frame count of video
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        flow_angles = flow_magnitudes = []
        histogram_temp_mean = np.zeros(bin_count)

        if frame is None:
            hist_magnitudes = hist_magnitudes / length
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create the old matrix to feed to LK, instead of goodFeaturesToTrack
        all_pixels = np.nonzero(frame_gray)[::-1]
        all_pixels = tuple(zip(*all_pixels))
        all_pixels = np.vstack(all_pixels).reshape(-1, 1, 2).astype("float32")

        # Calculate Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, all_pixels, None, **lk_params)
        all_pixels = all_pixels.reshape(height, width, 2)
        p1 = p1.reshape(height, width, 2)

        # Flow vector calculated by subtracting new pixels by old pixels
        flow = p1 - all_pixels

        # Update previous points for next iteration
        old_gray = frame_gray.copy()

    # Plot the temporal mean histogram
    # Increases the execution time, comment out when needed.
    # weights = np.ones_like(hist_magnitudes) / float(len(hist_magnitudes))
    # plt.hist(hist_magnitudes, bins=n_bins, weights=weights), plt.show()
    return hist_magnitudes

