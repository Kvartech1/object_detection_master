import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_yolo_model():
    # Load the YOLO model
    config_path = 'C:/Users/admin/Desktop/Machine Learning/object_detection_master/cfg/yolov3.cfg'
    weights_path = 'C:/Users/admin/Desktop/Machine Learning/object_detection_master/yolov3.weights'
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def detect_objects(image_path, net, output_layers):
    # Detect objects in the image
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Get bounding boxes and confidence scores
    boxes = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append((x, y, w, h))
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Filter out the detected objects and their labels
    detected_boxes = []
    detected_labels = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            detected_boxes.append((x, y, w, h))
            detected_labels.append(label)

    return detected_boxes, detected_labels

def get_dominant_colors(image_path, num_colors):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)

    # Perform k-means clustering to get the dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    colors = centers.tolist()

    return colors
    
    
def get_color_name(rgb_color):
    colors = {
        'Red': [255, 0, 0],
        'Green': [0, 255, 0],
        'Blue': [0, 0, 250],
        'Yellow': [255, 255, 0],
        'Violet': [238, 130, 238],
        'Orange': [255, 165, 0],
        'Black': [0, 0, 0],
        'White': [255, 255, 255],
        'Pink': [255, 192, 203],
        'Brown': [165, 42, 42]
}

    min_distance = float('inf')
    color_name = ""
    for name, color in colors.items():
        distance = np.sqrt(np.sum((np.array(color) - np.array(rgb_color)) ** 2))
        if distance < min_distance:
            min_distance = distance
            color_name = name
    return color_name

def draw_objects(image_path, boxes, labels):
    # Load the image
    image = cv2.imread(image_path)
    
    # Draw the bounding boxes and labels on the image
    for box, label in zip(boxes, labels):
        x, y, w, h = box
        color = (0, 255, 0)  # Green color for the rectangle and text
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Display the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def process_image(image_path):
    net, output_layers = load_yolo_model()
    boxes, labels = detect_objects(image_path, net, output_layers)
    draw_objects(image_path, boxes, labels)
    for label in labels:
        rgb_color = get_dominant_colors(image_path, 1)[0]
        color_name = get_color_name(rgb_color)
        print("Detected object:", label)
        print("Dominant color:", color_name)
        print()

# Process the image
image_path = 'dog-cycle-car.png'
process_image(image_path)








