import cv2
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward, fcluster    

img = cv2.imread('websocket/line.jpg')
img_canny = cv2.Canny(img, 50, 200, 3)

lines = cv2.HoughLines(img_canny, 1, 5* np.pi / 180, 150)

def find_parallel_lines(lines):

    lines_ = lines[:, 0, :]
    angle = lines_[:, 1]

    # Perform hierarchical clustering

    angle_ = angle[..., np.newaxis]
    y = pdist(angle_)
    Z = ward(y)
    cluster = fcluster(Z, 0.5, criterion='distance')

    parallel_lines = []
    for i in range(cluster.min(), cluster.max() + 1):
        temp = lines[np.where(cluster == i)]
        parallel_lines.append(temp.copy())
 
    return parallel_lines


cv2.imshow("lines", find_parallel_lines(lines)) 