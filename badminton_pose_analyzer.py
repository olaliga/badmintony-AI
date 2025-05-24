import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime
import os

class BadmintonPoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def calculate_angle(self, pointA, pointB, pointC):
        """計算三個點之間的角度"""
        vectorA = np.array([pointA[0] - pointB[0], pointA[1] - pointB[1]])
        vectorB = np.array([pointC[0] - pointB[0], pointC[1] - pointB[1]])
        
        dot = np.inner(vectorA, vectorB)
        mags = np.linalg.norm(vectorA) * np.linalg.norm(vectorB)
        cos = dot / mags
        
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        deg = np.rad2deg(rad)
        
        return int(deg)

    def calculate_vertical_angle(self, pointA, pointB):
        """計算與垂直線的角度"""
        vertical = np.array([0, -1])  # 垂直向下的向量
        vector = np.array([pointB[0] - pointA[0], pointB[1] - pointA[1]])
        
        dot = np.inner(vertical, vector)
        mags = np.linalg.norm(vertical) * np.linalg.norm(vector)
        cos = dot / mags
        
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        deg = np.rad2deg(rad)
        
        return int(deg)

    def extract_landmarks(self, landmarks):
        """提取關鍵點座標"""
        return {
            'left_ankle': [landmarks[27].x, landmarks[27].y, landmarks[27].z],
            'left_knee': [landmarks[25].x, landmarks[25].y, landmarks[25].z],
            'left_hip': [landmarks[23].x, landmarks[23].y, landmarks[23].z],
            'left_shoulder': [landmarks[11].x, landmarks[11].y, landmarks[11].z],
            'left_elbow': [landmarks[13].x, landmarks[13].y, landmarks[13].z],
            'left_wrist': [landmarks[15].x, landmarks[15].y, landmarks[15].z],
            'right_ankle': [landmarks[28].x, landmarks[28].y, landmarks[28].z],
            'right_knee': [landmarks[26].x, landmarks[26].y, landmarks[26].z],
            'right_hip': [landmarks[24].x, landmarks[24].y, landmarks[24].z],
            'right_shoulder': [landmarks[12].x, landmarks[12].y, landmarks[12].z],
            'right_elbow': [landmarks[14].x, landmarks[14].y, landmarks[14].z],
            'right_wrist': [landmarks[16].x, landmarks[16].y, landmarks[16].z]
        }

    def calculate_drive_measurements(self, landmarks):
        """計算平球姿勢的測量值"""
        points = self.extract_landmarks(landmarks)
        
        # 計算膝蓋彎曲角度
        left_knee_angle = self.calculate_angle(
            points['left_hip'],
            points['left_knee'],
            points['left_ankle']
        )
        right_knee_angle = self.calculate_angle(
            points['right_hip'],
            points['right_knee'],
            points['right_ankle']
        )
        
        # 計算上半身軸與法向量的角度
        upper_body_angle = self.calculate_vertical_angle(
            points['right_hip'],
            points['right_shoulder']
        )
        
        # 計算腋下角度
        left_armpit_angle = self.calculate_angle(
            points['left_shoulder'],
            points['left_elbow'],
            points['left_hip']
        )
        right_armpit_angle = self.calculate_angle(
            points['right_shoulder'],
            points['right_elbow'],
            points['right_hip']
        )
        
        # 計算手肘角度
        left_elbow_angle = self.calculate_angle(
            points['left_shoulder'],
            points['left_elbow'],
            points['left_wrist']
        )
        right_elbow_angle = self.calculate_angle(
            points['right_shoulder'],
            points['right_elbow'],
            points['right_wrist']
        )
        
        # 計算手肘位置（相對於肩膀的相對位置）
        left_elbow_position = {
            'x': points['left_elbow'][0] - points['left_shoulder'][0],
            'y': points['left_elbow'][1] - points['left_shoulder'][1]
        }
        right_elbow_position = {
            'x': points['right_elbow'][0] - points['right_shoulder'][0],
            'y': points['right_elbow'][1] - points['right_shoulder'][1]
        }
        
        # 計算上半身軸的位置（相對於髖部的相對位置）
        upper_body_position = {
            'x': points['right_shoulder'][0] - points['right_hip'][0],
            'y': points['right_shoulder'][1] - points['right_hip'][1]
        }
        
        return {
            'angles': {
                'left_knee_angle': left_knee_angle,
                'right_knee_angle': right_knee_angle,
                'upper_body_angle': upper_body_angle,
                'left_armpit_angle': left_armpit_angle,
                'right_armpit_angle': right_armpit_angle,
                'left_elbow_angle': left_elbow_angle,
                'right_elbow_angle': right_elbow_angle
            },
            'positions': {
                'left_elbow': left_elbow_position,
                'right_elbow': right_elbow_position,
                'upper_body': upper_body_position
            }
        }

    def analyze_drive_pose(self, image_path):
        """分析平球姿勢"""
        # 檢查檔案是否存在
        if not os.path.exists(image_path):
            raise ValueError(f"找不到圖片檔案：{image_path}")
            
        # 使用 numpy 讀取圖片（支援中文路徑）
        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"無法讀取圖片：{image_path}，請確認圖片格式是否正確（支援 jpg、jpeg、png）")
        except Exception as e:
            raise ValueError(f"讀取圖片時發生錯誤：{str(e)}")
        
        # 轉換顏色空間
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 進行姿勢檢測
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            raise ValueError("無法在圖片中檢測到人體姿勢，請確認圖片中有人物且姿勢清晰可見")
        
        # 計算測量值
        measurements = self.calculate_drive_measurements(results.pose_landmarks.landmark)
        
        # 在圖片上繪製骨架
        annotated_image = image.copy()
        self.mp_draw.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # # 在圖片上顯示測量值
        # y_position = 30
        
        # # 顯示角度
        # cv2.putText(annotated_image, "角度測量：", (10, y_position),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # y_position += 30
        
        # for angle_name, angle_value in measurements['angles'].items():
        #     text = f"{angle_name}: {angle_value}°"
        #     cv2.putText(annotated_image, text, (10, y_position),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #     y_position += 30
        
        # # 顯示位置
        # y_position += 20
        # cv2.putText(annotated_image, "位置測量：", (10, y_position),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # y_position += 30
        
        # for pos_name, pos_value in measurements['positions'].items():
        #     text = f"{pos_name}: x={pos_value['x']:.2f}, y={pos_value['y']:.2f}"
        #     cv2.putText(annotated_image, text, (10, y_position),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #     y_position += 30
        
        # 儲存結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 儲存標記後的圖片
        output_image_path = os.path.join('results', f'drive_pose_{timestamp}.jpg')
        cv2.imwrite(output_image_path, annotated_image)
        
        # 儲存測量數據到CSV
        df_angles = pd.DataFrame([measurements['angles']])
        df_positions = pd.DataFrame([measurements['positions']])
        
        csv_angles_path = os.path.join('results', f'drive_angles_{timestamp}.csv')
        csv_positions_path = os.path.join('results', f'drive_positions_{timestamp}.csv')
        
        df_angles.to_csv(csv_angles_path, index=False)
        df_positions.to_csv(csv_positions_path, index=False)
        
        return {
            'measurements': measurements,
            'annotated_image_path': output_image_path,
            'csv_angles_path': csv_angles_path,
            'csv_positions_path': csv_positions_path
        }

def main():
    # 建立results資料夾（如果不存在）
    os.makedirs('results', exist_ok=True)
    
    # 初始化分析器
    analyzer = BadmintonPoseAnalyzer()
    
    # 請使用者輸入圖片路徑
    image_path = input("請輸入平球姿勢圖片路徑：")
    
    try:
        # 分析圖片
        results = analyzer.analyze_drive_pose(image_path)
        
        print("\n分析完成！")
        print(f"標記後的圖片已儲存至：{results['annotated_image_path']}")
        print(f"角度數據已儲存至：{results['csv_angles_path']}")
        print(f"位置數據已儲存至：{results['csv_positions_path']}")
        
        # 顯示角度數據
        print("\n角度測量：")
        for angle_name, angle_value in results['measurements']['angles'].items():
            print(f"{angle_name}: {angle_value}°")
            
        # 顯示位置數據
        print("\n位置測量：")
        for pos_name, pos_value in results['measurements']['positions'].items():
            print(f"{pos_name}: x={pos_value['x']:.2f}, y={pos_value['y']:.2f}")
            
    except Exception as e:
        print(f"錯誤：{str(e)}")

if __name__ == "__main__":
    main() 