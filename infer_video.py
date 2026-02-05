import cv2
from ultralytics import YOLO

# 1. 加载最新的 YOLO26 Pose 预训练模型
# 模型会自动从 Ultralytics 官网下载
model = YOLO("./weights/yolo26n-pose.pt") 

# 2. 接入 USB 摄像头
# 参数 0 为默认摄像头，如果有多个摄像头可尝试 1, 2...
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率（可选，根据摄像头性能调整）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: 无法打开摄像头")
    exit()

print("正在启动 YOLO26 Pose 实时监控... 按 'q' 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. 执行推理
    # stream=True: 内存占用更低，适合持续的视频流
    # imgsz=640: 推理尺寸，可根据需求调小以换取更高 FPS
    results = model.predict(source=frame, stream=True, show=False)

    for r in results:
        # 4. 实时渲染骨架
        # plot() 会自动绘制 17 个关键点和人体骨骼连接线
        annotated_frame = r.plot()

        # 5. 显示窗口
        cv2.imshow("Ultralytics YOLO26 Real-Time Pose", annotated_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 资源释放
cap.release()
cv2.destroyAllWindows()