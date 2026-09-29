"""
part1_config.py
===============
Configuration Class สำหรับ YOLO Garbage Detection Project

วิธีใช้งาน:
    from part1_config import YOLOGarbageConfig
    config = YOLOGarbageConfig()
    config.display_config()
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path


class YOLOGarbageConfig:
    """
    ตั้งค่าพารามิเตอร์สำหรับ YOLO Object Detection
    
    ใช้สำหรับ Real-time Garbage Detection
    """
    
    def __init__(self):
        # ====================================
        # 🎯 1. โมเดล YOLO
        # ====================================
        
        # เลือกเวอร์ชัน YOLO
        self.MODEL_SIZE = 'yolo11n'  # n, s, m, l, x
        # PRETRAINED ใช้สำหรับ detection model เก่า (ไม่ใช้ใน classification)
        self.PRETRAINED = 'runs/detect/runs/garbage_detection_yolov8n/weights/best.pt'

        # ====================================
        # 📊 2. ข้อมูลและภาพ
        # ====================================

        self.IMG_SIZE = 640
        self.BATCH_SIZE = 16
        self.NUM_CLASSES = 10
        
        # ชื่อประเภทขยะ
        self.CLASS_NAMES = [
            'battery',
            'biological',
            'cardboard',
            'clothes',
            'glass',
            'metal',
            'paper',
            'plastic',
            'shoes',
            'trash'
        ]
        
        # สี bounding box (BGR)
        self.CLASS_COLORS = {
            'battery': (0, 0, 255),        # แดง
            'biological': (0, 128, 0),     # เขียวเข้ม
            'cardboard': (139, 69, 19),    # น้ำตาล
            'clothes': (255, 105, 180),    # ชมพู
            'glass': (0, 255, 255),        # ฟ้าอมเขียว
            'metal': (192, 192, 192),      # เงิน
            'paper': (255, 255, 255),      # ขาว
            'plastic': (0, 0, 255),        # น้ำเงิน
            'shoes': (255, 165, 0),        # ส้ม
            'trash': (0, 0, 0)             # ดำ
        }
        
        # ====================================
        # 🚀 3. การเทรน
        # ====================================

        self.EPOCHS = 150
        self.PATIENCE = 30
        self.LEARNING_RATE = 0.001       # ปรับลงเพราะใช้ AdamW
        self.OPTIMIZER = 'AdamW'         # ดีกว่า SGD สำหรับ classification
        self.LABEL_SMOOTHING = 0.1       # ป้องกัน overconfidence
        self.COS_LR = True               # cosine learning rate decay

        # ====================================
        # 🎨 4. Data Augmentation
        # ====================================

        self.MOSAIC = 1.0
        self.DEGREES = 45.0              # ↑ เพิ่มจาก 25 (ขยะถ่ายได้ทุกมุม)
        self.TRANSLATE = 0.2
        self.SCALE = 0.7
        self.FLIPLR = 0.5
        self.FLIPUD = 0.0
        self.HSV_H = 0.03                # ↑ เพิ่มจาก 0.02 (แสงไฟต่างกัน)
        self.HSV_S = 0.8
        self.HSV_V = 0.7                 # ↑ เพิ่มจาก 0.5 (กลางคืน-กลางวัน)
        self.MIXUP = 0.15
        self.ERASING = 0.4               # สุ่มลบส่วนภาพ เพิ่ม robustness
        
        # ====================================
        # 🎥 5. Real-time Detection
        # ====================================
        
        self.CONF_THRESHOLD = 0.25
        self.IOU_THRESHOLD = 0.45
        self.TARGET_FPS = 30
        self.SHOW_BOXES = True
        self.SHOW_LABELS = True
        self.SHOW_CONF = True
        self.BOX_THICKNESS = 2
        
        # ====================================
        # 📁 6. Paths
        # ====================================
        
        self.DATA_DIR = 'garbage_dataset_yolo'
        self.IMAGES_DIR = os.path.join(self.DATA_DIR, 'images')
        self.LABELS_DIR = os.path.join(self.DATA_DIR, 'labels')
        
        self.TRAIN_IMAGES = os.path.join(self.IMAGES_DIR, 'train')
        self.VAL_IMAGES = os.path.join(self.IMAGES_DIR, 'val')
        self.TRAIN_LABELS = os.path.join(self.LABELS_DIR, 'train')
        self.VAL_LABELS = os.path.join(self.LABELS_DIR, 'val')
        
        self.DATA_YAML = os.path.join(self.DATA_DIR, 'data.yaml')
        
        # ====================================
        # 📦 7. Classification
        # ====================================

        self.CLS_DATA_DIR = 'garbage_dataset_cls'
        self.CLS_MODEL = 'yolo11n-cls.pt'

        self.RUNS_DIR = 'runs'
        self.MODEL_DIR = 'models'
        self.RESULTS_DIR = 'results'
        
        # สร้างโฟลเดอร์
        self._create_directories()
    
    def _create_directories(self):
        """สร้างโครงสร้างโฟลเดอร์"""
        directories = [
            self.TRAIN_IMAGES,
            self.VAL_IMAGES,
            self.TRAIN_LABELS,
            self.VAL_LABELS,
            self.MODEL_DIR,
            self.RESULTS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_data_yaml(self):
        """สร้างไฟล์ data.yaml สำหรับ YOLO"""
        data_yaml_content = {
            'path': os.path.abspath(self.DATA_DIR),
            'train': 'images/train',
            'val': 'images/val',
            'nc': self.NUM_CLASSES,
            'names': self.CLASS_NAMES
        }
        
        with open(self.DATA_YAML, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False, 
                     allow_unicode=True)
        
        print(f"✅ สร้างไฟล์ data.yaml ที่: {self.DATA_YAML}")
        
        print("\n📄 เนื้อหาไฟล์ data.yaml:")
        print("-" * 40)
        with open(self.DATA_YAML, 'r', encoding='utf-8') as f:
            print(f.read())
        print("-" * 40)
    
    def display_config(self):
        """แสดงการตั้งค่าทั้งหมด"""
        print("\n" + "="*70)
        print("⚙️  YOLO GARBAGE DETECTION - CONFIGURATION")
        print("="*70)
        
        print("\n🎯 โมเดล YOLO:")
        print(f"  • เวอร์ชัน: {self.MODEL_SIZE}")
        print(f"  • Pre-trained: {self.PRETRAINED}")
        
        print("\n📊 ข้อมูล:")
        print(f"  • ขนาดภาพ: {self.IMG_SIZE}x{self.IMG_SIZE}")
        print(f"  • Batch Size: {self.BATCH_SIZE}")
        print(f"  • จำนวนประเภท: {self.NUM_CLASSES}")
        print(f"  • ประเภทขยะ: {', '.join(self.CLASS_NAMES)}")
        
        print("\n🚀 การเทรน:")
        print(f"  • Epochs: {self.EPOCHS}")
        print(f"  • Patience: {self.PATIENCE}")
        print(f"  • Learning Rate: {self.LEARNING_RATE}")
        print(f"  • Optimizer: {self.OPTIMIZER}")
        
        print("\n🎨 Data Augmentation:")
        print(f"  • Mosaic: {self.MOSAIC}")
        print(f"  • Rotation: ±{self.DEGREES}°")
        print(f"  • Translation: ±{self.TRANSLATE * 100}%")
        print(f"  • Scale: ±{self.SCALE * 100}%")
        print(f"  • Flip LR: {self.FLIPLR * 100}%")
        
        print("\n🎥 Real-time Detection:")
        print(f"  • Confidence: {self.CONF_THRESHOLD}")
        print(f"  • IOU: {self.IOU_THRESHOLD}")
        print(f"  • Target FPS: {self.TARGET_FPS}")
        
        print("\n📁 Directories:")
        print(f"  • Data: {self.DATA_DIR}")
        print(f"  • YAML: {self.DATA_YAML}")
        
        print("\n" + "="*70)
    
    def save_config(self):
        """บันทึกการตั้งค่าเป็น JSON"""
        config_dict = {
            'model_size': self.MODEL_SIZE,
            'img_size': self.IMG_SIZE,
            'batch_size': self.BATCH_SIZE,
            'num_classes': self.NUM_CLASSES,
            'class_names': self.CLASS_NAMES,
            'epochs': self.EPOCHS,
            'learning_rate': self.LEARNING_RATE,
            'conf_threshold': self.CONF_THRESHOLD,
            'iou_threshold': self.IOU_THRESHOLD,
            'target_fps': self.TARGET_FPS
        }
        
        config_path = os.path.join(self.RESULTS_DIR, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
        print(f"💾 บันทึก config ที่: {config_path}")
    
    def get_model_info(self):
        """แสดงข้อมูลโมเดล YOLO"""
        print("\n📊 เปรียบเทียบโมเดล YOLOv11:")
        print("-" * 70)
        print(f"{'Model':<15} {'Size':<10} {'Speed':<10} {'mAP':<10} {'Params':<15}")
        print("-" * 70)
        print(f"{'YOLO11n':<15} {'5.4MB':<10} {'⚡⚡⚡':<10} {'39.5':<10} {'2.6M':<15}")
        print(f"{'YOLO11s':<15} {'19.2MB':<10} {'⚡⚡':<10} {'47.0':<10} {'9.4M':<15}")
        print(f"{'YOLO11m':<15} {'40.5MB':<10} {'⚡':<10} {'51.5':<10} {'20.1M':<15}")
        print(f"{'YOLO11l':<15} {'51.4MB':<10} {'⚡':<10} {'53.4':<10} {'25.3M':<15}")
        print(f"{'YOLO11x':<15} {'114.6MB':<10} {'⚡':<10} {'54.7':<10} {'56.9M':<15}")
        print("-" * 70)
        print(f"\n✅ ปัจจุบันใช้: {self.MODEL_SIZE.upper()}")


# =============================================
# ทดสอบการทำงาน (ถ้ารันไฟล์นี้โดยตรง)
# =============================================

if __name__ == "__main__":
    print("🗑️ PART 1: CONFIGURATION\n")
    
    # สร้าง config
    config = YOLOGarbageConfig()
    
    # แสดงการตั้งค่า
    config.display_config()
    
    # สร้าง data.yaml
    config.create_data_yaml()
    
    # บันทึก config
    config.save_config()
    
    # แสดงข้อมูลโมเดล
    config.get_model_info()
    
    print("\n✅ Part 1 เสร็จสมบูรณ์!")
    print("\n📝 ขั้นตอนถัดไป:")
    print("   python part2_auto_annotation.py")