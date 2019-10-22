# Code from https://github.com/abewley/sort
"""
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}


    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import time
import argparse
import configparser
import cv2

# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from utils.metrics import iou

from align.detector import Face_Detector
from tracker import Tracker

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    # matched_indices = linear_assignment(-iou_matrix)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.concatenate((np.expand_dims(row_ind, axis=1), np.expand_dims(col_ind, axis=1)), axis=1)
    # matched_indices = linear_sum_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, unmatched_detections, unmatched_trackers


class Sort(object):
    def __init__(self, max_age=10000, min_hits=3):
        """
        Sets key parameters for SORT
        """

        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def reset(self):
        self.trackers = []
        self.frame_count = 0

    def update_detection(self, frame, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        # trks = np.zeros((len(self.trackers), 5))
        trks = []
        to_del = []

        self.frame_count += 1

        # get predicted locations from existing trackers.
        if len(self.trackers) > 0:
            for t in range(len(self.trackers)):
                pos = self.trackers[t].get_state(frame)
                trks.append([pos[0], pos[1], pos[2], pos[3], 0])
                if (np.any(np.isnan(pos))):
                    to_del.append(t)
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                # print('matched detection')
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d[0]], frame)
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # print('unmatched detection')
            trk = Tracker()
            trk.update(dets[i], frame)
            self.trackers.append(trk)

        # delete lost trackers
        for t, trk in reversed(list(enumerate(self.trackers))):
            if (t in unmatched_trks) and (self.trackers[t].time_since_update > self.max_age):
                self.trackers.pop(t)

    def track(self, frame):

        ret = []
        self.frame_count += 1

        for i, trk in reversed(list(enumerate(self.trackers))):

            d = trk.get_state(frame)

            ret.append(np.concatenate((d, [trk.id])))
            # remove dead track
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if (len(ret) > 0):
            return ret
        return np.empty((0, 5))




class FAST_DT():

    def __init__(self,
                 device='cpu',
                 tracker_max_age=10,
                 detection_per_frame=1):

        # create instance of the detector
        self.detector = Face_Detector(device)
        # create instance of the SORT tracker
        self.tracker = Sort(max_age=tracker_max_age)

        self.detection_per_frame = detection_per_frame
        self.frame_count = self.detection_per_frame
        self.dets_elapsed_time = 0.0

    def reset(self):
        self.frame_count = self.detection_per_frame
        self.tracker.reset()

    def predict(self, frame):

        # Launch a detection at each [detection_per_frame]
        if self.frame_count >= self.detection_per_frame:

            # Get detections bbx
            dets, landmarks = self.detector.detect_faces(frame)

            if len(dets) > 0:
                self.tracker.update_detection(frame, dets)

            # Reset the frame counter
            self.frame_count = 0

        # Track objects at each frame
        bbx = self.tracker.track(frame)

        self.frame_count += 1

        return bbx
