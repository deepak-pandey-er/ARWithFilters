"""
Extract cricket pitch centerline / crease markers using morphological ops + Hough line detection.
Returns list of line segments and candidate endpoints (image points for wickets).
"""

import cv2
import numpy as np

class PitchExtractor:
    def __init__(self):
        pass

    def extract_centerline(self, frame, mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use mask to focus on field; invert mask to look for pitch (non-grass)
        pitch_mask = cv2.bitwise_not(mask)
        # morphological cleaning
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_OPEN, k)
        # combine with edges
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, pitch_mask)
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, threshold=40, minLineLength=60, maxLineGap=40)
        segments = []
        if lines is not None:
            for l in lines[:,0,:]:
                segments.append(((int(l[0]),int(l[1])), (int(l[2]),int(l[3]))))
        # heuristics: find roughly vertical long line as pitch centerline
        if segments:
            lengths = [np.hypot(x2-x1, y2-y1) for ((x1,y1),(x2,y2)) in segments]
            idx = np.argmax(lengths)
            main = segments[idx]
            # pick endpoints as wicket candidates
            p1, p2 = main
            pitch_points = [p1, p2]
        else:
            pitch_points = []
        return segments, pitch_points