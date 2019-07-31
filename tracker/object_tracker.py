import time
import numpy as np
from .tracker_model import KCFTracker

tracker_count = 0


class Tracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """

    def __init__(self, tracker_type='kcf'):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model

        self.tracker = None

        if (tracker_type == 'kcf'):

            self.tracker = KCFTracker(True, False, True)  # hog, fixed_window, multiscale
        else:
            raise ValueError('Tracker type {} not supported.'.format(tracker_type))

        global tracker_count

        self.id = tracker_count
        tracker_count += 1
        self.time_since_update = 0
        self.hits = 0


    def update(self, bbox, frame):
        """
        Updates the state vector with observed bbox.
        """
        bboxNew=[]
        # print("before")
        # print(bbox)
        width = bbox[2]-bbox[0]
        height = bbox[3] -bbox[1]

        bboxNew.append(bbox[0])
        bboxNew.append(bbox[1])
        bboxNew.append(width)
        bboxNew.append(height)
        # print("after")
        # print(bboxNew)
        self.time_since_update = 0
        self.hits += 1
        self.tracker.init(bboxNew, frame)

    def get_state(self,frame):
        """
        Returns the current bounding box estimate.
        """
        self.time_since_update +=  1
        frame_height, frame_width, frame_channel = frame.shape

        bb = self.tracker.update(frame)

        return np.array([max(bb[0], 0),
                         max(bb[1], 0),
                         min(bb[0] + bb[2], frame_width),
                         min(bb[1] + bb[3], frame_height)])
