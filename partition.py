from ultralytics import YOLO
import cv2
import numpy as np
import aligb
from feature_comparison import get_features, get_imgs
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

price_list={
    'QQ糖':100,
    '香爆脆':200,
    '脆升升':18,
    '呀土豆':50,
    '乐事':60,
    '瓜子':80,
    '百岁山':20,
    '可口可乐':50,
    '苏打水':88,
    '娃哈哈矿泉水':90,
    '娃哈哈苏打水':100,
    '雪碧':200,
    '雀巢咖啡':300,
    '好丽友':250,
    '好多鱼':200,
    '百醇草莓味':200,
    '奥利奥饼干':300,
    '奥利奥可可脆卷':500,
    '百醇红酒味':200,
    '黑罐苏打水':400,
    '香草味苏打':200,
    '可口可乐罐装':500,
    '芬达':600,
    '美年达':600,
    '雪碧罐装':500,
    '玉米肠':100,
    '未知商品（特征库为空）':0,
"未知商品类别":0,
    'Q帝':0
}
def partition_img(frame, results, boxes):
    # Visualize the results on the frame
    segmented_images = []
    for idx, result in enumerate(results):
        frame_copy = np.copy(frame)
        if result.masks is not None:
            for idx, data in enumerate(result.masks.data):
                contours = data.detach().cpu().numpy() * 255
                mask = contours.astype(np.uint8)
                h, w, _ = frame_copy.shape
                mask_img = cv2.resize(mask, (w, h))
                mask = np.expand_dims(mask_img, axis=-1)
                img_seg = frame_copy & mask
                x1, y1, x2, y2 = boxes[idx][:4]
                img_seg = img_seg[int(y1):int(y2), int(x1):int(x2)]
                h, w, _ = img_seg.shape
                max_side = np.maximum(h, w)
                bg = np.zeros((max_side, max_side, 3), dtype=np.uint8)
                bg[(max_side - h) // 2:(max_side - h) // 2 + h, (max_side - w) // 2:(max_side - w) // 2 + w] = img_seg
                segmented_images.append(bg)

    # 返回所有分割后的图像，如果没有则返回None
    return segmented_images[0] if segmented_images else None


class Image_segmentation:
    def __init__(self):
        self.model = YOLO(r'D:\projects\PythonProject\smart supermarket\best_q.pt')
        self.font_path = r"C:\Windows\Fonts\STSONG.TTF"  # 替换为你的字体路径
        self.font_size = 20
        self.font_color = (255, 255, 0)  # 黄色
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        self.output_scale = 0.5
        self.store_labels = {}

    def video_partition(self, video_path, output_interval=20):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0  # 帧计数器
        output_count = 0  # 输出帧计数器（用于命名保存的文件）

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break  # 视频结束，退出循环

            # 调整帧大小
            new_width = int(frame.shape[1] * self.output_scale)
            new_height = int(frame.shape[0] * self.output_scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # 仅在指定间隔处理帧（例如每20帧）
            if frame_count % output_interval == 0:
                # 运行模型推理
                results = self.model.track(frame, persist=True,retina_masks=True)
                boxes = results[0].boxes.data.cpu().numpy()

                # 获取分割后的图像
                rotated_frame = partition_img(frame, results, boxes)

                # 确保rotated_frame不为None
                if rotated_frame is not None:
                    # 绘制标注
                    annotated_frame = self.draw(frame, results[0], rotated_frame, boxes)

                    # 显示当前帧
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)

                    # 可选：保存帧到文件
                    # cv2.imwrite(f"output_{output_count}.jpg", annotated_frame)
                    output_count += 1

            frame_count += 1  # 更新总帧数

            # 按 'q' 键提前退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def enroll_video(self, video_path, label):
        cap = cv2.VideoCapture(video_path)
        imgs = []
        frame_count = 0  # 记录当前帧号
        skip_frames = 30  # 每隔 30 帧取 1 帧
        big_labels = []

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # 仅处理每隔 skip_frames 帧
                if frame_count % skip_frames == 0:
                    # 调整大小
                    new_width = int(frame.shape[1] * self.output_scale)
                    new_height = int(frame.shape[0] * self.output_scale)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

                    # 模型处理
                    try:
                        results = self.model.track(frame, persist=True)
                        boxes = results[0].boxes.data.cpu().numpy()

                        # 处理每个检测结果
                        for box in boxes:
                            if len(box) == 7:
                                x1, y1, x2, y2, track_id, conf, cls_id = box[:7]
                                cls_name = results[0].names[int(cls_id)]
                                big_labels.append(cls_name)

                        # 获取分割后的图像
                        rotated_frame = partition_img(frame, results, boxes)
                        if rotated_frame is not None:
                            imgs.append(rotated_frame)

                    except Exception as e:
                        print(f"Processing error at frame {frame_count}: {e}")
                        continue

                frame_count += 1  # 更新帧计数

            print(f'视频处理完成，共提取 {len(imgs)} 帧（每隔 {skip_frames} 帧）')
            if not imgs:
                return

            # 显示部分（简化版，因为原代码中的交互式显示可能不适合批量处理）
            for i, img in enumerate(imgs[:5]):  # 只显示前5张作为示例
                cv2.imshow(f'Sample Image {i + 1}', img)
            cv2.waitKey(0)

        finally:
            cap.release()
            cv2.destroyAllWindows()

        # 统计最常见的类别
        if big_labels:
            counter = Counter(big_labels)
            big_label = counter.most_common(1)[0][0]
            get_features(imgs, label, big_label)

    def draw(self, frame, result, rotated_frame, boxes):
        total_merchandise= {}
        for box in boxes:
            if len(box) == 7:
                x1, y1, x2, y2, track_id, conf, cls_id = box[:7]
                cls_name = result.names[int(cls_id)]

                # 绘制边界框
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # 获取中文标签
                if track_id in self.store_labels:
                    label_name = self.store_labels[track_id]
                else:
                    # 确保rotated_frame不为None
                    if rotated_frame is not None:
                        label_name = get_imgs(rotated_frame, cls_name)
                        self.store_labels[track_id] = label_name
                    else:
                        label_name = "未知"

                # 构造标签文本
                label = f"{int(track_id)}--{label_name}--{conf:.2f}"
                total_merchandise[label_name] = total_merchandise.get(label_name, 0) + 1


                # 调用优化后的中文绘制方法
                frame = self.put_chinese_text(frame, label, (int(x1), int(y1) - 20))
        print(total_merchandise)
        sum =0
        for name,price in total_merchandise.items():
            sum+=int(price_list[name])*int(price)
        print(sum)
        return frame

    def put_chinese_text(self, frame, text, position):
        """优化后的中文绘制方法（使用预加载的字体）"""
        try:
            # 转换图像格式（BGR → RGB）
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # 使用预加载的字体绘制文本
            draw.text(position, text, font=self.font, fill=self.font_color)

            # 转换回 OpenCV 格式（RGB → BGR）
            return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[Error] 绘制中文失败: {e}")
            return frame  # 失败时返回原图


if __name__ == '__main__':
    a = Image_segmentation()
    # a.enroll_video(r'D:\projects\PythonProject\smart supermarket\img\text\2.mp4', '苏打水')
    a.video_partition(r'D:\projects\PythonProject\smart supermarket\img\text\4.mp4')