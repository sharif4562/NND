import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

print("Loading model...")
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
model = hub.load(model_url)
print("Model loaded successfully")

def detect_cars(image_path, threshold=0.5):
    
    if isinstance(image_path, str):
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
            img = np.array(img)
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_path
    
    original_img = img.copy()
    height, width = img.shape[:2]
    
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    scale = 640 / max(height, width)
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
        height, width = img.shape[:2]
    
    img_tensor = tf.convert_to_tensor(img)
    img_tensor = img_tensor[tf.newaxis, ...]
    
    detections = model(img_tensor)
    
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
    
    car_mask = (classes == 3) & (scores > threshold)
    car_boxes = boxes[car_mask]
    car_scores = scores[car_mask]
    
    detected_cars = []
    for i, (box, score) in enumerate(zip(car_boxes, car_scores)):
        y1, x1, y2, x2 = box
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)
        
        if scale < 1:
            x1 = int(x1 / scale)
            x2 = int(x2 / scale)
            y1 = int(y1 / scale)
            y2 = int(y2 / scale)
            original_h, original_w = original_img.shape[:2]
            x1 = max(0, min(x1, original_w))
            x2 = max(0, min(x2, original_w))
            y1 = max(0, min(y1, original_h))
            y2 = max(0, min(y2, original_h))
        
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        label = f"Car: {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(original_img, (x1, y1-25), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(original_img, label, (x1, y1-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        detected_cars.append({
            'position': (x1, y1, x2, y2),
            'confidence': score
        })
    
    return original_img, detected_cars

def analyze_parking_gate(image_path):
    print(f"\nAnalyzing image: {image_path}")
    print("-" * 50)
    
    try:
        result_img, cars = detect_cars(image_path, threshold=0.4)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.title(f"Vehicle Detection at Parking Gate\nTotal Cars Detected: {len(cars)}", 
                  fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"\nPARKING GATE STATUS:")
        print(f"   Cars detected: {len(cars)}")
        
        if len(cars) == 0:
            print("   Gate: CLOSED - No vehicles")
        elif len(cars) == 1:
            print("   Gate: OPENING - Single vehicle detected")
            print(f"      Confidence: {cars[0]['confidence']:.2%}")
        elif len(cars) <= 3:
            print(f"   Gate: OPENING - {len(cars)} vehicles detected")
            for i, car in enumerate(cars, 1):
                print(f"      Vehicle {i}: {car['confidence']:.2%} confidence")
        else:
            print(f"   Gate: QUEUE SYSTEM ACTIVE - {len(cars)} vehicles waiting")
            print("      Opening barrier for sequential entry")
        
        print("-" * 50)
        return len(cars), result_img
        
    except Exception as e:
        print(f"Error: {e}")
        return 0, None

def create_sample_image():
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200
    
    cv2.rectangle(img, (0, 400), (800, 600), (100, 100, 100), -1)
    cv2.line(img, (0, 400), (800, 400), (255, 255, 255), 2)
    
    cv2.rectangle(img, (350, 250), (450, 400), (150, 150, 150), -1)
    cv2.rectangle(img, (350, 250), (450, 400), (0, 0, 0), 3)
    cv2.putText(img, "GATE", (370, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cars = [
        {"box": (100, 430, 220, 550), "color": (0, 0, 255)},
        {"box": (300, 450, 420, 570), "color": (0, 255, 0)},
        {"box": (550, 440, 670, 560), "color": (255, 0, 0)},
        {"box": (700, 460, 780, 550), "color": (0, 255, 255)},
    ]
    
    for car in cars:
        x1, y1, x2, y2 = car["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), car["color"], -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.circle(img, (x1+20, y2-5), 12, (0, 0, 0), -1)
        cv2.circle(img, (x2-20, y2-5), 12, (0, 0, 0), -1)
        cv2.circle(img, (x1+20, y1+5), 12, (0, 0, 0), -1)
        cv2.circle(img, (x2-20, y1+5), 12, (0, 0, 0), -1)
    
    cv2.putText(img, "PARKING GATE SIMULATION", (250, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

if __name__ == "__main__":
    print("\nSMART PARKING SYSTEM")
    print("=" * 50)
    
    print("\nOptions:")
    print("1. Enter image path (local file or URL)")
    print("2. Use sample image with simulated cars")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        image_path = input("Enter image path or URL: ").strip()
        if not image_path:
            print("No path provided. Using sample image...")
            sample_img = create_sample_image()
            car_count, result = analyze_parking_gate(sample_img)
        else:
            car_count, result = analyze_parking_gate(image_path)
    else:
        print("Generating sample parking gate image...")
        sample_img = create_sample_image()
        car_count, result = analyze_parking_gate(sample_img)
    
    if result is not None:
        print(f"\nDetection complete! Found {car_count} vehicle(s)")
    else:
        print("\nDetection failed. Please check your image and try again.")
