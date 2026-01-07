from ultralytics import solutions

inf = solutions.Inference(
    model = "train/weights/best.pt"
)

inf.inference()