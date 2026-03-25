import cv2

# 打开原始视频文件
video_path = r"F:\Yolov9\ultralytics-main\ultralytics-main\mycode\1.mp4"
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# 获取原始视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 设置新视频的分辨率和编解码器
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
print("%d,%d",frame_width,frame_height)
# 创建新视频文件
out = cv2.VideoWriter('1_new.mp4', fourcc, 30, (frame_width, frame_height))

frame_count = 0
success, frame = cap.read()

second = 4.8658
# 读取并写入每秒的一帧
while success:
    # 每秒的第一帧
    if frame_count % int(fps*second) == 0:
        out.write(frame)
        print(f"Writing frame {frame_count}")

    # 读取下一帧
    success, frame = cap.read()
    frame_count += 1

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete.")