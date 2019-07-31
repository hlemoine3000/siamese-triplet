
import cv2


class Video_Reader():

    def __init__(self, video_path):

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        assert self.cap.isOpened(), "Error opening video stream or file"

    def __del__(self):
        self.cap.release()

    def reset(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        assert self.cap.isOpened(), "Error opening video stream or file"

        return 0

    def get_frame(self):

        ret = False
        frame = None

        if (not self.cap.isOpened()):
            print('Video completed.')
            return (ret, frame)

        ret, frame = self.cap.read()

        return (ret, frame)

