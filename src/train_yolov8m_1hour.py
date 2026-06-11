from ultralytics import YOLO


def main():
    model = YOLO("yolov8m.pt")
    model.train(
        data="src/yolo_dataset/data.yaml",
        epochs=80,
        imgsz=960,
        batch=4,
        device=0,
        project="runs/yolo_coin",
        name="yolov8m_960_1hour",
        exist_ok=True,
        workers=0,
        patience=20,
    )


if __name__ == "__main__":
    main()
