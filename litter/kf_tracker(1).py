"""
追踪：
检测第n帧的图像 dets
{"box": box, "cls": cls, "conf": conf, "track_id": track_id}
old_trackers = []
new_trackers = []
trackers = old_trackers + new_trackers

情况1：n-1帧目标进行iou计算 iou>阈值
更新已经存在的跟踪器，更新位置
old_tracker

情况2: iou小于阈值的就是新的
dic = {"box": box, "cls": cls, "conf": conf, "track_id": track_id}
new_trackers.append(dic)

trackers = old_trackers + new_trackers
---------------------------------------
IOU:
- 只要有遮挡 就会出问题
- 匹配
--------------

"""
import time

import cv2
import numpy as np
from motrackers import CentroidKF_Tracker
from ultralytics import YOLO


class Tracker:
    def __init__(self):
        self.model = YOLO(r'/runs/segment/train3/weights/best.pt')
        self.names = {0: 'bag', 1: 'bottle', 2: 'box', 3: 'can'}
        # 存储追踪器
        # 追踪器的标识

        self.track_id = 0
        # 实例化跟踪器
        self.tracker = CentroidKF_Tracker(max_lost=60)

    def predict(self, frame):
        result = self.model(frame)[0]
        boxes = result.boxes  # Boxes object for bbox outputs
        boxes = boxes.cpu().numpy()  # convert to numpy array

        dets = []  # 检测结果
        # 参考：https://docs.ultralytics.com/modes/predict/#boxes
        # 遍历每个框
        for box in boxes.data:
            xin, ymin, xmax, ymax = box[:4]  # left, top, right, bottom
            conf, class_id = box[4:]  # confidence, class
            if class_id not in self.names:
                continue
            if conf>0.9:
                dets.append({"box": (xin, ymin, xmax, ymax),
                             "conf": conf,
                             "cls": class_id})
        return dets

    def bbox_iou(self, box, bbox):
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        l_x = np.maximum(box[0], bbox[0])
        l_y = np.maximum(box[1], bbox[1])
        r_x = np.minimum(box[2], bbox[2])
        r_y = np.minimum(box[3], bbox[3])
        w = np.maximum(r_x - l_x, 0)
        h = np.maximum(r_y - l_y, 0)

        inter_area = w * h
        val = inter_area / (box_area + bbox_area - inter_area)
        return val

    def process_kf_format(self, dets):
        '''
        处理格式：https://github.com/adipandas/multi-object-tracker
        '''
        bboxes, confidences, class_ids = [], [], []
        for det in dets:
            bbox = det['box']
            conf = det['conf']
            class_id = det['cls']
            x_min = np.minimum(bbox[0], bbox[2])
            x_max = np.maximum(bbox[0], bbox[2])
            y_min = np.minimum(bbox[1], bbox[3])
            y_max = np.maximum(bbox[1], bbox[3])
            w = x_max - x_min
            h = y_max - y_min
            if conf > 0.9:
                bboxes.append([x_min, y_min, w, h])
                confidences.append(conf)
                class_ids.append(class_id)

        bboxes = np.array(bboxes).astype('int')
        confidences = np.array(confidences)
        class_ids = np.array(class_ids).astype('int')

        return bboxes, confidences, class_ids

    def create_trackers(self, frame):
        # 检测
        dets = self.predict(frame)
        # 处理一下格式
        bboxes, confidences, class_ids = self.process_kf_format(dets)
        # 更新跟踪器
        tracks = self.tracker.update(bboxes, confidences, class_ids)
        # 遍历跟踪器

        for idx,track in enumerate(tracks):
                # 分别是：(<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>, <class_id> )
                frame_num, id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
                # 绘制跟踪框

                cv2.rectangle(frame, (bb_left, bb_top), (bb_left + bb_width, bb_top + bb_height),
                              (0, 255, 0), 1)
                # 绘制id
                cv2.putText(frame, str(id), (bb_left, bb_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, str(class_ids[idx]), (bb_left+ bb_width, bb_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


    def run(self):
        cap = cv2.VideoCapture(r'/4.mp4')
        st = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 缩放至720p
            end = time.time()
            fps = 1 / (end - st)
            st = end
            cv2.putText(frame,
                        str(fps),
                        org=(100, 100),
                        fontFace=cv2.FONT_ITALIC,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=1)
            # 创建跟踪器
            self.create_trackers(frame)
            # 绘制效果
            # self.draw_box(frame)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tracker = Tracker()
    tracker.run()
