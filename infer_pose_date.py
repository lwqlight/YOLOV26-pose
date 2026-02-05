import cv2
from ultralytics import YOLO

# ==========================================
# 1. 关键点映射表 (COCO 标准 17点)
# ==========================================
KEYPOINT_MAP = {
    0: "鼻子", 1: "左眼", 2: "右眼", 3: "左耳", 4: "右耳",
    5: "左肩", 6: "右肩", 7: "左肘", 8: "右肘", 9: "左腕", 10: "右腕",
    11: "左胯", 12: "右胯", 13: "左膝", 14: "右膝", 15: "左踝", 16: "右踝"
}

def main():
    # 2. 加载 YOLO26 Pose 模型
    # 如果本地没有，会自动下载
    model = YOLO("./weights/yolo26s-pose.pt")

    # 3. 打开 USB 摄像头 (索引 0)
    cap = cv2.VideoCapture(0)
    
    # 设置合适的分辨率以保证帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("=== YOLO26 Pose 启动成功 ===")
    print("按 'q' 退出程序")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. 执行推理
        # stream=True 模式更适合视频流，减少内存堆积
        results = model.predict(frame, stream=True, show=False, verbose=False)

        # 这里 results 是个生成器，我们处理当前帧的结果
        for r in results:
            # --- A. 画面渲染 ---
            # plot() 自动绘制骨架连线和关键点
            annotated_frame = r.plot()
            cv2.imshow("YOLO26 Pose Estimation", annotated_frame)

            # --- B. 数据输出 ---
            # r.keypoints 包含了当前帧所有目标的坐标
            if r.keypoints is not None:
                # 获取 xy 坐标，并转为 numpy 数组以便处理
                # shape: (N, 17, 2)，N 是人数
                keypoints_xy = r.keypoints.xy.cpu().numpy()
                
                # 获取置信度 (可见性)
                keypoints_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

                # 只有当画面中有人时才打印分割线，避免刷屏
                if len(keypoints_xy) > 0:
                    print(f"\n--- 当前帧检测到 {len(keypoints_xy)} 人 ---")

                # 遍历每一个人 (Target)
                for person_idx, kpts in enumerate(keypoints_xy):
                    print(f"【目标 ID: {person_idx + 1}】")
                    
                    # 遍历这个人的 17 个关键点
                    for kp_idx, (x, y) in enumerate(kpts):
                        # 获取置信度，如果置信度太低（比如遮挡），可以选择不输出
                        conf = keypoints_conf[person_idx][kp_idx] if keypoints_conf is not None else 1.0
                        
                        # 过滤掉未检测到的点 (通常坐标为 0,0 或置信度极低)
                        if conf > 0.5 and (x > 0 or y > 0):
                            part_name = KEYPOINT_MAP.get(kp_idx, f"Point {kp_idx}")
                            # 格式化输出：部位名称: (x, y)
                            print(f"  - {part_name:<3}: ({x:.1f}, {y:.1f})")

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()