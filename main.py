"""
===========================
@Author  : Pranjal Rai
@Version: 1.0    12/07/2020
Retinal vessel segmentation 
and diameter estimation
===========================
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import diameter_calc


img = cv2.imread("PATH")



diameter_calc.diameter(img)