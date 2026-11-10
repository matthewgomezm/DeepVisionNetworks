from ultralytics import YOLO 

source = "../data/portrait.jpeg"

model = YOLO("yolo26n.pt")

results = model(source)

for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.show()
    result.save(filename = "result.jpeg")