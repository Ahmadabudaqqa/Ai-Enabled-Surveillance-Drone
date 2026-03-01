from ultralytics import YOLO


def main():
    # Faster settings for MX450: fewer epochs, smaller image, AMP enabled
    model = YOLO('yolov8n-cls.pt')
    model.train(
        data='activity_dataset',
        epochs=10,
        imgsz=160,
        batch=8,
        device=0,
        workers=4,
        amp=True,
        project='runs/train',
        name='activity_cls_gpu_fast'
    )


if __name__ == '__main__':
    main()
