import cv2
import os

def video_to_frames(video_path, output_folder, interval=1, img_format="jpg"):
    """
    将视频抽帧保存为图片（直接运行版，无需命令行参数）
    :param video_path: 视频文件路径（如 "D:/videos/test.mp4"）
    :param output_folder: 输出文件夹（如 "D:/output_frames"）
    :param interval: 抽帧间隔（每隔多少帧抽1帧，默认1）
    :param img_format: 图片格式（"jpg" 或 "png"，默认 "jpg"）
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: 无法打开视频文件！")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    print(f"视频信息: {fps} FPS, 共 {total_frames} 帧")

    # 抽帧并保存
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 interval 帧保存一次
        if frame_count % interval == 0:
            output_path = os.path.join(output_folder, f"baisuisan_{saved_count:06d}.{img_format}")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"抽帧完成！共保存 {saved_count} 张图片到 {output_folder}")

if __name__ == "__main__":
    # 直接在代码里指定参数（无需命令行输入）
    video_path = r"D:\projects\PythonProject\smart supermarket\baisuishan.mp4"  # 视频路径（修改成你的）
    output_folder = r"D:\projects\PythonProject\smart supermarket\img\baisuisan"  # 输出文件夹（修改成你的）
    interval = 30 # 每隔 30 帧抽 1 帧（可修改）
    img_format = "jpg"  # 图片格式（jpg/png）

    # 调用抽帧函数
    video_to_frames(video_path, output_folder, interval, img_format)