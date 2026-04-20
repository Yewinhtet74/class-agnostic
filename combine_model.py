import os

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any, Optional
from smallcnn import SmallImageCNN

class CombinedModel:
    def __init__(
        self,
        yolo_model_path: str = 'yolov8n.pt',
        small_cnn_path: str = 'small_cnn.pth',
        classifier_input_size: int = (64*64),
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """
        Initialize the combined model.
        
        Args:
            yolo_model_path: Path to YOLO model weights
            small_cnn_path: Path to small CNN model weights
            device: Device to run inference on
            confidence_threshold: Confidence threshold for YOLO detections
            iou_threshold: IoU threshold for NMS in YOLO
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize YOLO detector
        self.yolo = YOLO(yolo_model_path)
        
        # Initialize small CNN
        self.smallcnn = SmallImageCNN()
        
        # Load state dict properly
        state_dict = torch.load(small_cnn_path, map_location=device)
        missing_keys, unexpected_keys = self.smallcnn.load_state_dict(state_dict, strict=False)
        self.smallcnn.to(device)
        self.smallcnn.eval()
        
        # Image preprocessing for CNN
        self.transform = transforms.Compose([
            transforms.Resize(classifier_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def preprocess_boxes(
        self, 
        image: np.ndarray, 
        boxes: List[List[float]]
    ) -> torch.Tensor:
        """
        Extract and preprocess bounding boxes for CNN classification.
        
        Args:
            image: Original image (numpy array in BGR format)
            boxes: List of bounding boxes in [x1, y1, x2, y2] format
            
        Returns:
            Tensor of preprocessed box images
        """
        box_images = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure box coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Extract box region
            box_region = image[y1:y2, x1:x2]
            
            if box_region.size == 0:
                continue
                
            # Convert BGR to RGB
            box_rgb = cv2.cvtColor(box_region, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and preprocess
            pil_image = Image.fromarray(box_rgb)
            transformed = self.transform(pil_image)
            box_images.append(transformed)
            
        if box_images:
            return torch.stack(box_images)
        return torch.tensor([])
    
    def draw_and_save(
        self,
        image: np.ndarray,
        boxes: List[List[float]],
        classes: List[int],
        confidences: List[float],
        save_path: str
    ):
        """
        Draw bounding boxes on the image and save it.
        
        Args:
            image: Original image (numpy array in BGR format)
            boxes: List of bounding boxes in [x1, y1, x2, y2] format
            classes: List of class indices for each box
            confidences: List of confidence scores for each box
            save_path: Path to save the annotated image
        """
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"Class {cls}: {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # # Draw label background
            # label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            # cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
            #              (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # # Draw label text
            # cv2.putText(image, label, (x1, y1 - 5), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.imwrite(save_path, image)
    
    def detect_and_classify(
        self, 
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Perform detection and classification on a single image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of dictionaries containing detection and classification results
        """
        # Run YOLO detection
        results = self.yolo(
            image, 
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Extract bounding boxes from YOLO results
        boxes = []
        yolo_classes = []
        yolo_confidences = []
        
        if len(results) > 0:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes_xyxy, classes, confs):
                boxes.append(box.tolist())
                yolo_classes.append(int(cls))
                yolo_confidences.append(float(conf))
        # print(boxes)
        # print(yolo_classes)
        # print(yolo_confidences)
        # Draw each box
        # self.draw_and_save(image, boxes, yolo_classes, yolo_confidences, 'yolo_detections.jpg')
        results = []
    
        # Process each box
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Crop image
            cropped_np = image[y1:y2, x1:x2]
            
            if cropped_np.size == 0:
                print(f"Warning: Empty crop for box {i}")
                continue
            
            # Save crop if requested
            if 0:
                crop_path = os.path.join('test', f'crop_{i}.jpg')
                cv2.imwrite(crop_path, cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR))
                print(f"Saved: {crop_path} (size: {cropped_np.shape})")
            
            # Convert numpy array to PIL Image
            cropped_pil = Image.fromarray(cropped_np)
            
            
            # Transform to tensor
            cropped_tensor = self.transform(cropped_pil)
            
            # Add batch dimension and move to device
            cropped_tensor = cropped_tensor.unsqueeze(0)  # Add batch dim
            cropped_tensor = cropped_tensor.to(self.device)  # Move to device
            
            # Classify
            with torch.no_grad():
                outputs = self.smallcnn(cropped_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Store results
            result = {
                'box_index': i,
                'box_coordinates': box,
                'crop_array': cropped_np,
                'crop_tensor': cropped_tensor,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy() 
            }
            results.append(result)
            
            print(f"Box {i}: Predicted class = {predicted_class}, Confidence = {confidence:.4f}")
        
        return results
    
        return 0
        
        # Preprocess boxes for CNN classification
        if boxes:
            box_tensors = self.preprocess_boxes(image, boxes)
            
            # Run CNN classification
            if len(box_tensors) > 0:
                box_tensors = box_tensors.to(self.device)
                with torch.no_grad():
                    cnn_outputs = self.cnn(box_tensors)
                    cnn_probs = F.softmax(cnn_outputs, dim=1)
                    cnn_classes = torch.argmax(cnn_probs, dim=1)
                    cnn_confidences = torch.max(cnn_probs, dim=1)[0]
                
                cnn_classes = cnn_classes.cpu().numpy()
                cnn_confidences = cnn_confidences.cpu().numpy()
            else:
                cnn_classes = []
                cnn_confidences = []
        else:
            cnn_classes = []
            cnn_confidences = []
        
        # Combine results
        results_list = []
        for i, box in enumerate(boxes):
            result = {
                'bbox': box,
                'yolo_class': yolo_classes[i] if i < len(yolo_classes) else None,
                'yolo_confidence': yolo_confidences[i] if i < len(yolo_confidences) else None,
                'cnn_class': int(cnn_classes[i]) if i < len(cnn_classes) else None,
                'cnn_confidence': float(cnn_confidences[i]) if i < len(cnn_confidences) else None,
            }
            results_list.append(result)
        
        return results_list
    
    def detect_and_classify_batch(
        self, 
        images: List[np.ndarray],
        batch_size: int = 32
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform detection and classification on a batch of images.
        
        Args:
            images: List of input images
            batch_size: Batch size for CNN inference
            
        Returns:
            List of results for each image
        """
        all_results = []
        
        # Process each image individually for detection (YOLO handles batching internally)
        for image in images:
            results = self.detect_and_classify(image)
            all_results.append(results)
            
        return all_results
    
    def visualize_results(
        self,
        image: np.ndarray,
        results: List[Dict[str, Any]],
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detection and classification results on the image.
        
        Args:
            image: Original image
            results: Results from detect_and_classify
            class_names: List of class names for display
            save_path: Path to save visualization
            
        Returns:
            Image with visualized results
        """
        vis_image = image.copy()
        
        for result in results:
            x1, y1, x2, y2 = map(int, result['bbox'])
            
            # Get class name
            if class_names and result['cnn_class'] is not None:
                class_name = class_names[result['cnn_class']]
            else:
                class_name = f"Class_{result['cnn_class']}"
            
            # Create label
            label = f"{class_name}: {result['cnn_confidence']:.2f}"
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            
        return vis_image
    


# Initialize combined model
combined_model = CombinedModel(
    yolo_model_path= 'runs/detect/yolo11n-visdrone-agnostic/training/weights/best.pt',
    small_cnn_path= 'smallcnn_logs/1/best_model.pth',
    classifier_input_size= (64*64),
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    confidence_threshold = 0.1,
    iou_threshold = 0.45
)

# Example: Process a single image
image = cv2.imread('VisDrone-org/images/test/0000006_00159_d_0000001.jpg')

results = combined_model.detect_and_classify(image)

# Visualize results
# class_names = ['class_0', 'class_1', ...]  # Your class names
# vis_image = combined_model.visualize_results(
#     image, results, class_names, 'output.jpg'
# )

# # Print results
# for i, result in enumerate(results):
#     print(f"Object {i}:")
#     print(f"  Bounding Box: {result['bbox']}")
#     print(f"  YOLO Class: {result['yolo_class']}")
#     print(f"  YOLO Confidence: {result['yolo_confidence']:.3f}")
#     print(f"  CNN Class: {result['cnn_class']}")
#     print(f"  CNN Confidence: {result['cnn_confidence']:.3f}")
