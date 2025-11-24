# Train_Model_Facemask_YoloV8s 
**Author:** PhCtrlZ  
**Project:** Huấn luyện mô hình phát hiện khẩu trang bằng YOLOv8s

## Giới thiệu  
Dự án này thực hiện huấn luyện mô hình YOLOv8s để phát hiện **Mask** và **No Mask** trên ảnh hoặc video. Mục tiêu là tạo một mô hình nhẹ, chính xác và có thể triển khai trên máy tính cá nhân hoặc thiết bị IoT/Edge.

##  Cấu trúc thư mục  
├── data.yaml
├── main.py
├── train/
├── valid/
├── runs/
│ └── detect/
├── test.jpg
└── README.dataset.txt

##  Yêu cầu môi trường  
- Python 3.x  
- torch  
- ultralytics  
- opencv-python  
- numpy  

Cài đặt nhanh:
```bash
pip install -r requirements.txt
python train.py --data data.yaml --model yolov8s.pt --epochs 50 --batch 16
python main.py --source 0 --weights runs/train/.../best.pt --img 640
