"""
Termux 兼容版国服 Clash Royale YOLOv8 实时识别
专为 Android Termux 环境优化，无需 APK 编译
使用 QOFFSCREEN 避免 X server 依赖
"""
import cv2
import numpy as np
import os
import time
import sys

# 设置 Qt 离屏平台（避免需要 X server）
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# 尝试导入 YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[错误] 缺少 ultralytics，请运行: pip install ultralytics opencv-python numpy")

# 尝试导入 OpenCV（核心库）
import cv2

# -----------------------------
# 🧠 模型加载
# -----------------------------
MODEL_PATH = 'best_cn.pt' if os.path.exists('best_cn.pt') else 'best.pt'
if YOLO_AVAILABLE:
    model = YOLO(MODEL_PATH)
    print(f"[国服模式] 加载模型: {MODEL_PATH}")
else:
    print("[错误] 未安装 YOLO")
    sys.exit(1)

# 国服中文卡牌名称映射
CN_CARD_NAMES = {
    0: '飞龙宝宝', 1: '骑士', 2: '弓箭手', 3: '哥布林',
    4: '骷髅兵', 5: '巨人', 6: '火球', 7: '飞龙宝宝',
    8: '石头人', 9: '毒液', 10: '骷髅军团', 11: '公主',
    12: '骑士', 13: '弓箭手', 14: '飞龙宝宝', 15: '骷髅兵',
    16: '气球', 17: '石人', 18: '龙宝宝', 19: '骷髅兵',
}

# 屏幕配置
SCREEN_MON = {'top': 100, 'left': 0, 'width': 1920, 'height': 1080}

def draw_detections(frame, results):
    """绘制检测框和中文标签"""
    annotated = frame.copy()
    if results[0].boxes is None:
        return annotated
    
    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        if conf > 0.4:
            label = f"{CN_CARD_NAMES.get(cls_id, f'单位_{cls_id}')} ({conf:.2f})"
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # 绘制背景半透明
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            # 绘制矩形背景
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 0, 0), -1)
            # 绘制文字
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 绘制检测框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 显示置信度文字
            cv2.putText(annotated, f"信:{conf:.1%}", (x2 - 50, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return annotated

def test_mode():
    """测试模式 - 创建测试图像并识别"""
    print("[测试模式] 创建测试图像...")
    
    # 创建一个黑色背景测试图
    test_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 在图上添加一些文字说明
    cv2.putText(test_img, '国服CR实时识别测试', (100, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(test_img, 'Termux 版 - 离线模式', (200, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 添加模拟的检测框区域
    # 模拟检测到 3 个目标
    for i in range(3):
        y = 200 + i * 150
        # 绘制模拟的检测矩形
        cv2.rectangle(test_img, (50, y), (350, y + 100), (255, 0, 0), 3)
        cv2.putText(test_img, f'测试目标_{i+1}', (60, y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 运行 YOLO 识别
    print("[识别中...]")
    results = model.predict(test_img, verbose=False, conf=0.4, iou=0.45)
    
    # 绘制检测结果
    print("[处理结果...]")
    annotated = draw_detections(test_img, results)
    
    # 保存结果
    output_path = '测试识别结果.jpg'
    cv2.imwrite(output_path, annotated)
    print(f"[完成] 结果已保存到: {output_path}")
    print(f"[文件大小] {os.path.getsize(output_path)} 字节")
    
    # 显示简要信息（离屏模式下只保存）
    try:
        # 尝试使用 imshow，离屏模式下会自动处理
        cv2.imshow('国服识别结果', annotated)
        print("[显示] 窗口已创建（离屏模式）")
        cv2.waitKey(1)  # Brief wait
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[提示] 显示被跳过，结果已保存: {e}")

def image_mode(image_path):
    """图片识别模式"""
    if not os.path.exists(image_path):
        print(f"[错误] 文件不存在: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"[错误] 无法读取图片: {image_path}")
        return
    
    print(f"[识别中] {image_path}")
    results = model.predict(img, verbose=False, conf=0.4, iou=0.45)
    annotated = draw_detections(img, results)
    
    # 保存结果
    output_path = f"识别结果_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"[完成] 结果已保存到: {output_path}")
    print(f"[文件大小] {os.path.getsize(output_path)} 字节")
    
    # 尝试显示
    try:
        cv2.imshow('国服识别结果', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("[提示] 结果已保存（显示模式不可用）")

def main():
    """主函数"""
    print("=" * 50)
    print("国服 Clash Royale YOLOv8 实时识别 (Termux 版)")
    print("=" * 50)
    print("[Qt平台] 已设置离屏模式: offscreen")
    print()
    
    if len(sys.argv) < 2:
        print("\n[使用说明]")
        print("1. 测试模式:   python3 termux_version.py test")
        print("2. 图片识别:   python3 termux_version.py image path/to/image.jpg")
        print("\n[提示]")
        print("- 测试模式无需任何参数，直接运行即可")
        print("- 结果将保存为 JPG 文件在当前目录")
        print("- 检测框和中文标签将绘制在图像上")
        print()
        print("[快速开始]")
        print("直接运行: python3 termux_version.py test")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == 'test':
        test_mode()
    elif mode == 'image' and len(sys.argv) > 2:
        image_mode(sys.argv[2])
    else:
        print(f"[错误] 未知模式: {mode}")
        print("可用模式: test, image")

if __name__ == '__main__':
    main()