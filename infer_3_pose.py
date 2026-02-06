import cv2
import math
import numpy as np
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self):
        # 关键点索引 (COCO格式)
        self.KEYPOINTS = {
            'nose': 0, 'left_shoulder': 5, 'right_shoulder': 6,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }

    def calculate_angle(self, p1, p2, p3):
        """计算三点夹角 (p2为顶点)"""
        # 过滤掉无效点 (0,0)
        if any(c == 0 for c in [*p1, *p2, *p3]):
            return 0
            
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        # 计算向量 BA 和 BC
        ba = a - b
        bc = c - b
        
        # 计算余弦夹角
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def determine_pose(self, kpts):
        """
        根据关键点坐标判断姿态
        kpts: numpy array (17, 2)
        """
        kp = {k: kpts[v] for k, v in self.KEYPOINTS.items()}
        
        # 1. 计算【躯干倾斜度】 (判断躺下)
        # 取肩膀中心和臀部中心
        mid_shoulder = (kp['left_shoulder'] + kp['right_shoulder']) / 2
        mid_hip = (kp['left_hip'] + kp['right_hip']) / 2
        
        # 计算躯干向量 (dy, dx) 注意：图像坐标系y向下
        dy = mid_hip[1] - mid_shoulder[1]
        dx = mid_hip[0] - mid_shoulder[0]
        
        # 躯干与垂直方向的夹角 (0度为直立，90度为水平)
        # atan2 返回弧度，abs(dx/dy) 越大说明越水平
        if dy != 0:
            inclination = abs(math.degrees(math.atan2(dx, dy)))
        else:
            inclination = 90

        # === 判定逻辑 1: 躺下 ===
        # 如果躯干倾斜角 > 60度 (比较水平)，判定为躺下
        if inclination > 60:
            return "Lying Down"

        # 2. 计算【膝盖角度】 (判断坐/站)
        # 分别计算左腿和右腿的角度
        l_angle = self.calculate_angle(kp['left_hip'], kp['left_knee'], kp['left_ankle'])
        r_angle = self.calculate_angle(kp['right_hip'], kp['right_knee'], kp['right_ankle'])
        
        # 取较大的那个角度（避免侧身时一条腿被遮挡导致误判）
        # 或者取平均值，这里取最大值比较稳健（只要有一条腿直就是站）
        knee_angle = max(l_angle, r_angle)

        # === 判定逻辑 2 & 3: 站立 vs 坐立 ===
        # 阈值设定：腿部伸直通常 > 160度，坐着通常 < 120度。取 145度做分界线。
        if knee_angle > 145:
            return f"Standing | Angle: {int(knee_angle)}"
        else:
            return f"Sitting| Angle: {int(knee_angle)}"

# ================= 主程序 =================
def main():
    model = YOLO("./weights/yolo26n-pose.pt")
    estimator = PoseEstimator()
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    print("开始姿态识别... 按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 推理
        results = model.predict(frame, stream=True, show=False, verbose=False)

        for r in results:
            # 绘制骨架
            frame = r.plot()
            
            # 获取所有人的关键点数据 (N, 17, 2)
            if r.keypoints and r.keypoints.xy is not None:
                kpts_all = r.keypoints.xy.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy() # 获取人脸框用于定位文字

                for idx, kpts in enumerate(kpts_all):
                    # 调用姿态判断逻辑
                    pose_status = estimator.determine_pose(kpts)
                    
                    # 在头顶绘制文字
                    # 获取 bounding box 的左上角坐标
                    if len(boxes) > idx:
                        x1, y1, _, _ = boxes[idx]
                        cv2.putText(frame, pose_status, (int(x1), int(y1) + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Pose Classification", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()