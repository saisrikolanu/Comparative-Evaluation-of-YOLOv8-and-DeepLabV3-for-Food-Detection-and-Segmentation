# Comparative Evaluation of YOLOv8 and DeepLabV3+ for Food Detection and Segmentation

**Course Project**  
**Author:** Sai Sri Kolanu (50594437, saisriko)  
**University at Buffalo**  

---

## 📖 Overview
This project compares **semantic segmentation** and **object detection** methods for food image analysis using the **UECFood100 dataset**.  

- **Segmentation Model:** DeepLabV3+ with MobileNetV3 backbone (transfer learning).  
- **Detection Model:** YOLOv8n fine-tuned on the dataset.  
- Evaluation performed using **Mean IoU (mIoU)** for segmentation and **mAP@0.5 / mAP@[0.5:0.95]** for detection.  

---

## 📊 Dataset
- **Source:** [UECFood100 Dataset](http://foodcam.mobi/dataset100.html)  
- **Classes:** 100 Japanese food categories  
- **Splits:** 80% training, 20% validation  
- **Preprocessing:**
  - Resize images to **320×320**
  - Normalize with ImageNet mean/std
  - Masks resized with nearest-neighbor interpolation  

**Augmentations:**
- Segmentation: Random horizontal flips, random crops  
- Detection: Horizontal flips, random scaling (±10%)  

---

## ⚙️ Methods
### DeepLabV3+ (Semantic Segmentation)
- Backbone: MobileNetV3-Large (ImageNet pretrained)  
- Optimizer: AdamW, LR = 1e-4 → 1e-6 (cosine decay)  
- Loss: Weighted Cross-Entropy + Dice Loss  
- Epochs: 50, Batch Size: 8  
- Metric: **mIoU = 0.559**  
- Inference speed: ~25 ms/image  

### YOLOv8n (Object Detection)
- Base: YOLOv8n pretrained on COCO  
- Frozen early backbone layers during fine-tuning  
- Optimizer: AdamW (auto LR ~9.6e-5), weight decay 5e-4  
- Epochs: 10, Batch Size: 4  
- Losses: CIoU (bbox), BCE (objectness), Focal BCE (class)  
- Metrics:  
  - **mAP@0.5 = 0.327**  
  - **mAP@[0.5:0.95] = 0.248**  
- Inference speed: ~2 ms/image  

---

## 📈 Results
| Model        | Primary Metric | Score   | Inference Time |
|--------------|---------------|---------|----------------|
| DeepLabV3+   | mIoU          | 0.559   | ~25 ms/image   |
| YOLOv8n      | mAP@0.5       | 0.327   | ~2 ms/image    |
| YOLOv8n      | mAP@[0.5:0.95]| 0.248   | ~2 ms/image    |

- **DeepLabV3+** excels at pixel-level segmentation but is slower.  
- **YOLOv8n** is much faster, making it suitable for real-time applications, but with lower per-class accuracy.  

---

## 🛠️ Tech Stack
- **Python 3.x**
- **PyTorch 1.13.1**
- **segmentation_models_pytorch**
- **Ultralytics YOLOv8**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
- **TensorBoard** (logging)

---

## ▶️ How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/saisrikolanu/Comparative-Evaluation-of-YOLOv8-and-DeepLabV3-for-Food-Detection-and-Segmentation.git
   cd Comparative-Evaluation-of-YOLOv8-and-DeepLabV3-for-Food-Detection-and-Segmentation
