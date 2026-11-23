"""
国服专用 Clash Royale YOLOv8 识别脚本（无 GUI 模式）
支持：图片输入 / 摄像头 / 测试模式
"""
import cv2
import numpy as np
import os
import sys
import argparse
from ultralytics import YOLO

# -----------------------------
# 🧠 加载模型（国服版权重优先）
# -----------------------------
model_path = 'best_cn.pt' if os.path.exists('best_cn.pt') else 'best.pt'
model = YOLO(model_path)
print(f"[国服模式] 加载模型: {model_path}")

# -----------------------------
# 🖥️ 国服专属屏幕配置（无 GUI 模式用）
# -----------------------------
SCREEN_MON = {'top': 100, 'left': 0, 'width': 1920, 'height': 1080}

# 国服中文卡牌名称映射（可扩展）
CN_CARD_NAMES = {
    0: '飞龙宝宝',
    1: '骑士',
    2: '弓箭手',
    3: '哥布林',
    4: '骷髅兵',
    5: '巨人身材',
    6: '火球',
    7: '飞龙宝宝',
    8: '石头人',
    9: '毒液',
    10: '骷髅军团',
    11: '公主',
    12: '骑士',
    13: '弓箭手',
    14: '飞龙宝宝',
    15: '骷髅兵',
    16: '气球',
    17: '石人',
    18: '龙宝宝',
    19: '骷髅兵',
}

def draw_chinese_labels(frame, results):
    """在检测框上绘制中文名称"""
    annotated = frame.copy()
    if results[0].boxes is None:
        return annotated
    
    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        if cls_id in CN_CARD_NAMES:
            label = f"{CN_CARD_NAMES[cls_id]} ({conf:.2f})"
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # 绘制半透明背景
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制检测框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return annotated

def process_image(img):
    """处理单张图片"""
    print("[INFO] 处理图片...")
    results = model.predict(img, verbose=False, conf=0.4, iou=0.45)
    annotated = draw_chinese_labels(img, results)
    
    # 保存结果
    output_path = '国服识别结果.jpg'
    cv2.imwrite(output_path, annotated)
    print(f"[完成] 保存到: {output_path}")
    
    # 如果支持 GUI，显示窗口
    try:
        cv2.imshow('国服实时识别 (按任意键退出)', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("[提示] 无 GUI 环境，已保存到文件")
    
    return annotated

def process_camera():
    """摄像头实时识别（无 GUI 模式下用视频文件替代）"""
    print("[INFO] 摄像头模式需要 GUI 支持，请尝试图片模式")
    # 这里可以扩展为读取视频文件

def process_screen():
    """屏幕捕获模式（无 GUI 下不可用）"""
    print("[INFO] 屏幕捕获需要 GUI 支持，请尝试图片模式")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='国服 Clash Royale YOLOv8 识别')
    parser.add_argument('--image', type=str, help='输入图片路径')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    
    args = parser.parse_args()
    
    if args.image and os.path.exists(args.image):
        img = cv2.imread(args.image)
        if img is not None:
            process_image(img)
        else:
            print(f"[错误] 无法读取图片: {args.image}")
    elif args.test:
        # 测试模式：创建一个测试图像
        print("[测试模式] 创建测试图像...")
        test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(test_img, '国服识别测试', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        results = model.predict(test_img, verbose=False)
        annotated = draw_chinese_labels(test_img, results)
        cv2.imwrite('测试识别结果.jpg', annotated)
        print("[完成] 测试结果已保存到: 测试识别结果.jpg")
    else:
        print("[使用说明]")
        print("1. 图片识别: python3 main_cn.py --image path/to/image.jpg")
        print("2. 测试模式: python3 main_cn.py --test")
        print("3. 如果你有 best_cn.pt 权重文件，会自动使用它")
        print("\n[注意] 当前环境无 GUI，结果会保存到文件而非显示窗口")