from ultralytics import YOLO
import cv2
import os

if __name__ == "__main__":

    # ====== 2. LOAD MODEL SAU KHI TRAIN ======
    best_model_path = os.path.join("runs", "detect", "mask_detect", "weights", "best.pt")

    if not os.path.exists(best_model_path):
        print("⚠️ Không tìm thấy file best.pt! Hãy kiểm tra đường dẫn hoặc quá trình train có lỗi không.")
        exit()

    model = YOLO(best_model_path)
    print(f"✅ Model đã load: {best_model_path}")

    # ====== 3. TEST MODEL TRÊN ẢNH ======
    test_img = r"D:\NCKH\train\dev\test.jpg"  # ảnh test
    if os.path.exists(test_img):
        results = model.predict(source=test_img, show=True, conf=0.5)
    else:
        print(f"⚠️ Không tìm thấy file test.jpg để test ảnh!")

    # ====== 4. NHẬN DIỆN REAL-TIME QUA WEBCAM ======
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam!")
        exit()

    print("🎥 Đang mở webcam... Nhấn ESC để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được khung hình!")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # Nhấn ESC để thoát
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
