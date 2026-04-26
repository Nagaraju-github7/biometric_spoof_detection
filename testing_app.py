#!/usr/bin/env python3
"""
Advanced Fingerprint Spoof Detection UI with User Feedback System
Model selection, bulk testing, failure analysis, and user correction capabilities
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageOps
import time
import os
import shutil
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Import scanner module
try:
    from fingerprint_scanner import scanner_ui
    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Advanced Fingerprint Spoof Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .danger-box {
        background: #ef4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .warning-box {
        background: #f59e0b;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .info-box {
        background: #3b82f6;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 8px;
        margin: 0.25rem 0;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

class FingerprintResNet(nn.Module):
    """ResNet18 with Transfer Learning for Fingerprint Spoof Detection"""
    
    def __init__(self, num_classes=2, pretrained=False):
        super(FingerprintResNet, self).__init__()
        
        # Load ResNet18 architecture
        self.model = models.resnet18(pretrained=pretrained)
        
        # Get original features
        num_features = self.model.fc.in_features
        
        # Create flexible final layer that can handle both simple and sequential
        self.model.fc = nn.Sequential(
            nn.Flatten(),  # Flatten features
            nn.Linear(num_features, num_classes)  # Final classification
        )
        
        # Initialize new layers
        if not pretrained:
            nn.init.xavier_uniform_(self.model.fc[1].weight)
            nn.init.constant_(self.model.fc[1].bias, 0)
    
    def forward(self, x):
        return self.model(x)

class FingerprintResNetA600(nn.Module):
    """ResNet18 with Transfer Learning for Fingerprint Spoof Detection - Compatible with checkpoint"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(FingerprintResNetA600, self).__init__()
        
        # Load pretrained ResNet18 (exact match to Colab training)
        self.model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # IMPORTANT: Replace fc layer with exact structure from checkpoint
        num_features = self.model.fc.in_features
        
        # Replace final layer for binary classification (exact match to checkpoint)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)  # num_classes=2 for Fake/Live
        )
        
        # Initialize new layers (only if not loading from checkpoint)
        if pretrained:
            nn.init.xavier_uniform_(self.model.fc[1].weight)
            nn.init.constant_(self.model.fc[1].bias, 0)
        
        print(f"FingerprintResNetA600 initialized:")
        print(f"   - Output classes: {num_classes} (Fake/Live)")
        print(f"   - FC layer shape: {num_features} -> {num_classes}")
        print(f"   - Pretrained: {pretrained}")
    
    def forward(self, x):
        return self.model(x)

class WorkingModel(nn.Module):
    """Simple working model for testing"""
    
    def __init__(self, num_classes=2):
        super(WorkingModel, self).__init__()
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class FingerprintResNetCustom(nn.Module):
    """Custom ResNet for best_fingerprint_model.pth"""
    
    def __init__(self, num_classes=2):
        super(FingerprintResNetCustom, self).__init__()
        
        # Custom architecture based on checkpoint keys
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Custom layers based on checkpoint
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)  # Add another maxpool
        x = self.conv3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class FingerprintResNetNagaraju(nn.Module):
    """Fixed 4-layer ResNet18 for Nagaraju models - matches checkpoint exactly"""
    
    def __init__(self, num_classes=2):
        super(FingerprintResNetNagaraju, self).__init__()
        
        # Load ResNet18 architecture with ImageNet weights
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Get original features
        num_features = self.model.fc.in_features
        
        # Create the exact structure that matches the checkpoint:
        # FC.1: Linear(512, 512)
        # FC.2: BatchNorm1d(512) 
        # FC.5: Linear(512, 2)
        
        # First linear layer (FC.1)
        self.fc1_linear = nn.Linear(num_features, 512)
        
        # BatchNorm layer (FC.2) - only one BN layer
        self.fc_bn = nn.BatchNorm1d(512)
        
        # Final linear layer (FC.5)
        self.fc5_linear = nn.Linear(512, num_classes)
        
        # Activation and dropout
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)
        
        print(f"FingerprintResNetNagaraju initialized:")
        print(f"   Expected checkpoint keys: model.fc.1.*, model.fc.2.*, model.fc.5.*")
        print(f"   Final layer shape: 512 -> {num_classes}")
    
    def forward(self, x):
        # ResNet backbone
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Custom classifier (matching checkpoint structure)
        x = self.dropout1(x)
        x = self.fc1_linear(x)
        x = self.fc_bn(x)  # Only one BN layer
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc5_linear(x)
        
        return x
    
    def load_state_dict_custom(self, state_dict):
        """Custom state dict loading to match checkpoint structure"""
        # Load ResNet backbone
        backbone_dict = {}
        custom_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('model.fc.'):
                custom_dict[key] = value
            else:
                backbone_dict[key] = value
        
        # Load backbone
        self.model.load_state_dict(backbone_dict, strict=False)
        
        # Load custom layers
        # FC.1: Linear layer
        self.fc1_linear.weight.data = custom_dict['model.fc.1.weight']
        self.fc1_linear.bias.data = custom_dict['model.fc.1.bias']
        
        # FC.2: BatchNorm layer
        self.fc_bn.weight.data = custom_dict['model.fc.2.weight']
        self.fc_bn.bias.data = custom_dict['model.fc.2.bias']
        self.fc_bn.running_mean.data = custom_dict['model.fc.2.running_mean']
        self.fc_bn.running_var.data = custom_dict['model.fc.2.running_var']
        self.fc_bn.num_batches_tracked.data = custom_dict['model.fc.2.num_batches_tracked']
        
        # FC.5: Final linear layer
        self.fc5_linear.weight.data = custom_dict['model.fc.5.weight']
        self.fc5_linear.bias.data = custom_dict['model.fc.5.bias']

class FingerprintResNet25Layer(nn.Module):
    """2.5-layer ResNet18 for Nagaraju models"""
    
    def __init__(self, num_classes=2):
        super(FingerprintResNet25Layer, self).__init__()
        
        # Load ResNet18 architecture
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Get original features
        num_features = self.model.fc.in_features
        
        # 2.5-layer classifier: Dropout→Linear→BatchNorm→ReLU→Dropout→Linear
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class FingerprintResNetLegacy(nn.Module):
    """Legacy ResNet18 for older model files"""
    
    def __init__(self, num_classes=2):
        super(FingerprintResNetLegacy, self).__init__()
        
        # Load ResNet18 architecture
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Get original features
        num_features = self.model.fc.in_features
        
        # Legacy 2-layer classifier (for older models)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class FingerprintResNetEnhanced(nn.Module):
    """Enhanced ResNet18 with Advanced Architecture for Optimized Model"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(FingerprintResNetEnhanced, self).__init__()
        
        # Load ResNet18 architecture
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Get original features
        num_features = self.model.fc.in_features
        
        # Enhanced classifier matching the training script (3-layer)
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_fingerprint_model(model_path):
    """Load fingerprint spoof detection model with robust CPU/GPU compatibility"""
    try:
        print(f"Loading model: {model_path}")
        
        # Device detection and map_location setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load with proper map_location to handle CPU/GPU differences
        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            print(f"Model loaded with map_location='cpu'")
        except Exception as e:
            print(f"Error with map_location='cpu', trying default: {e}")
            checkpoint = torch.load(model_path, map_location=device)
            print(f"Model loaded with map_location='{device}'")
        
        # Extract state dict from checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"Found 'model_state_dict' in checkpoint")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"Found 'state_dict' in checkpoint")
            else:
                state_dict = checkpoint
                print(f"Using checkpoint as direct state_dict")
        else:
            state_dict = checkpoint
            print(f"Checkpoint is direct state_dict")
        
        # Analyze checkpoint structure
        print(f"Analyzing checkpoint structure...")
        fc_keys = [k for k in state_dict.keys() if 'fc' in k]
        print(f"   Found {len(fc_keys)} FC-related keys: {fc_keys[:5]}...")
        
        # Check model filename first to determine architecture
        model_filename = Path(model_path).name.lower()
        print(f"Model filename: {model_filename}")
        
        # Priority 1: Check for Nagaraju architecture in checkpoint structure
        fc_keys = [k for k in state_dict.keys() if 'fc' in k]
        has_nagaraju_keys = any(key in fc_keys for key in ['model.fc.2.weight', 'model.fc.5.weight'])
        
        if has_nagaraju_keys or 'nagaraju' in model_filename:
            # Use Nagaraju 4-layer model with custom loading
            print(f"Using FingerprintResNetNagaraju (4-layer classifier)")
            model = FingerprintResNetNagaraju(num_classes=2)
            model_type = "Nagaraju ResNet18 (4-layer classifier)"
            
            # Use custom loading method for Nagaraju models
            try:
                model.load_state_dict_custom(state_dict)
                print(f"Custom state dict loading successful")
            except Exception as e:
                print(f"Custom loading failed, trying standard: {e}")
                model.load_state_dict(state_dict, strict=False)
        elif any(keyword in model_filename for keyword in ['a600', 'colab', 'resnet18', 'final', 'best']):
            print(f"Using FingerprintResNetA600 (Colab-trained ResNet18)")
            model = FingerprintResNetA600(num_classes=2, pretrained=False)  # Ensure 2 classes
            model_type = "FingerprintResNetA600 (Colab-trained ResNet18 - 2 classes)"
        elif 'working_test_model' in model_filename:
            print(f"Using working test model")
            model = WorkingModel(num_classes=2)
            model_type = "Working Test Model (simple CNN)"
        elif 'best_fingerprint_model' in model_filename:
            print(f"Using custom model")
            model = FingerprintResNetCustom(num_classes=2)
            model_type = "Custom ResNet (simple classifier)"
        else:
            # Fallback to FingerprintResNetA600 with 2 classes
            print(f"Using default FingerprintResNetA600 (2 classes)")
            model = FingerprintResNetA600(num_classes=2, pretrained=False)
            model_type = "FingerprintResNetA600 (default - 2 classes)"
        
        # Verify model architecture before loading
        print(f"Verifying model architecture...")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Expected output classes: 2 (Fake/Live)")
        
        # Load state dict with error handling (skip for Nagaraju models)
        if not ('nagaraju' in model_filename and hasattr(model, 'load_state_dict_custom')):
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"State dict loaded successfully (strict=False)")
            except Exception as e:
                print(f"Error loading state dict: {e}")
                # Try with strict=True for more specific error
                try:
                    model.load_state_dict(state_dict, strict=True)
                    print(f"State dict loaded successfully (strict=True)")
                except Exception as e2:
                    print(f"Strict loading also failed: {e2}")
                    raise e2
        else:
            print(f"Custom loading already completed for Nagaraju model")
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Verify final layer
        if hasattr(model, 'model') and hasattr(model.model, 'fc'):
            fc_layer = model.model.fc
            if hasattr(fc_layer, '__iter__'):  # Sequential (FingerprintResNetA600)
                final_layer = fc_layer[-1]  # Get last layer in Sequential
                if hasattr(final_layer, 'out_features'):
                    out_features = final_layer.out_features
                else:
                    out_features = 'unknown'
            else:
                out_features = fc_layer.out_features if hasattr(fc_layer, 'out_features') else 'unknown'
            
            print(f"Final layer output features: {out_features}")
            if out_features == 2:
                print(f"Correctly configured for 2 classes (Fake/Live)")
            else:
                print(f"Warning: Expected 2 classes, got {out_features}")
        elif hasattr(model, 'fc5_linear'):  # For Nagaraju models
            out_features = model.fc5_linear.out_features
            print(f"Final layer output features: {out_features}")
            if out_features == 2:
                print(f"Correctly configured for 2 classes (Fake/Live)")
            else:
                print(f"Warning: Expected 2 classes, got {out_features}")
        elif hasattr(model, 'classifier'):  # For WorkingModel
            out_features = model.classifier.out_features
            print(f"Final layer output features: {out_features}")
            if out_features == 2:
                print(f"Correctly configured for 2 classes (Fake/Live)")
            else:
                print(f"Warning: Expected 2 classes, got {out_features}")
        else:
            print(f"Model structure verification skipped (different architecture)")
        
        print(f"Model loaded successfully!")
        print(f"   Architecture: {model_type}")
        print(f"   Device: {device}")
        print(f"   Model path: {model_path}")
        
        return model, device
            
    except Exception as e:
        print(f"CRITICAL ERROR loading model {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_available_models():
    """Scan models folder for available .pth files with priority for optimized model"""
    models_dir = Path("./models")
    if not models_dir.exists():
        return []
    
    model_files = list(models_dir.glob("*.pth"))
    model_paths = [str(f) for f in model_files]
    
    # Sort to prioritize the new optimized model
    optimized_models = [m for m in model_paths if "Nagaraju_Final_ResNet" in m]
    other_models = [m for m in model_paths if "Nagaraju_Final_ResNet" not in m]
    
    # Return optimized models first, then others
    return optimized_models + other_models

def get_test_dataset_structure():
    """Analyze test dataset structure"""
    test_dir = Path("./dataset/test")
    if not test_dir.exists():
        return {}
    
    structure = {}
    
    # Check if it has subdirectories (material folders)
    has_subdirs = any(d.is_dir() for d in test_dir.iterdir())
    
    if has_subdirs:
        # Material-based structure - check subdirectories
        for material_dir in test_dir.iterdir():
            if material_dir.is_dir():
                material_name = material_dir.name
                
                # Check if this material folder has subdirectories
                material_subdirs = [d for d in material_dir.iterdir() if d.is_dir()]
                
                if material_subdirs:
                    # Has material type subdirectories
                    structure[material_name] = {}
                    for subdir in material_subdirs:
                        image_files = list(subdir.glob("*.*"))
                        structure[material_name][subdir.name] = len(image_files)
                else:
                    # Direct image files in material folder
                    image_files = list(material_dir.glob("*.*"))
                    structure[material_name] = len(image_files)
    else:
        # Flat file structure - treat all files as one category
        image_files = list(test_dir.glob("*.*"))
        structure["Test Images"] = len(image_files)
    
    return structure

def create_failure_directories(model_name):
    """Create failure analysis directories"""
    test_dir = Path("./dataset/test")
    failures_dir = Path("./failures") / model_name
    
    if not test_dir.exists():
        return {}
    
    # Check if it has subdirectories (material folders)
    has_subdirs = any(d.is_dir() for d in test_dir.iterdir())
    
    if has_subdirs:
        # Material-based structure
        material_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        for material_dir in material_dirs:
            failure_path = failures_dir / material_dir.name
            failure_path.mkdir(parents=True, exist_ok=True)
        
        return {d.name: failures_dir / d.name for d in material_dirs}
    else:
        # Flat file structure - create single failure directory
        failure_path = failures_dir / "Test Images"
        failure_path.mkdir(parents=True, exist_ok=True)
        return {"Test Images": failure_path}

def preprocess_image(image):
    """Enhanced preprocessing with CLAHE for better ridge detection"""
    try:
        # Force RGB conversion for all inputs
        if isinstance(image, Image.Image):
            # Check and convert image mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            pil_image = image
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Try to convert BGR to RGB if cv2 is available
                try:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                except:
                    # Fallback: assume RGB
                    pil_image = Image.fromarray(image)
            else:
                # Assume RGB
                pil_image = Image.fromarray(image)
        else:
            return None
        
        # Try enhanced preprocessing with CLAHE if cv2 is available
        try:
            # Convert PIL to OpenCV format for CLAHE
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel (lightness)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l, a, b])
            cv_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Convert back to PIL
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        except:
            # Fallback: use original PIL image without CLAHE enhancement
            pass
        
        # Apply transforms matching Colab training script exactly
        transform = transforms.Compose([
            transforms.Resize(256),  # Match Colab validation transforms
            transforms.CenterCrop(224),  # Match Colab validation transforms
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization (match Colab)
                std=[0.229, 0.224, 0.225]   # ImageNet normalization (match Colab)
            )
        ])
        
        image_tensor = transform(pil_image)
        
        # Validate tensor shape
        if image_tensor.shape != (3, 224, 224):
            print(f"⚠️ Unexpected tensor shape: {image_tensor.shape}, expected (3, 224, 224)")
        
        return image_tensor
        
    except Exception as e:
        print(f"❌ Error in preprocess_image: {e}")
        return None

def run_single_inference(model, device, image):
    """Run inference on a single image"""
    try:
        image_tensor = preprocess_image(image)
        if image_tensor is None:
            return None, 0.0
        
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()
        
    except Exception as e:
        return None, 0.0

def save_user_feedback(image, image_path, prediction, user_label, confidence, material_type=None):
    """Save user feedback with actual image for model improvement"""
    try:
        feedback_data = {
            'image_path': str(image_path),
            'model_prediction': prediction,
            'user_label': user_label,
            'confidence': confidence,
            'material_type': material_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create feedback directory structure
        feedback_dir = Path("./user_feedback")
        feedback_dir.mkdir(exist_ok=True)
        
        # Create subdirectories based on user label and material
        user_label_name = "Live" if user_label == 1 else "Spoof"
        
        if material_type:
            # If material is specified, organize by material type
            image_dir = feedback_dir / user_label_name / material_type
        else:
            # Otherwise organize by user label only
            image_dir = feedback_dir / user_label_name
        
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the actual image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"{Path(image_path).stem}_{timestamp}.png"
        image_save_path = image_dir / image_filename
        
        # Convert image to RGB if needed and save
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(image_save_path, 'PNG')
        
        # Update feedback data with saved image path
        feedback_data['saved_image_path'] = str(image_save_path)
        
        # Save feedback JSON
        feedback_file = feedback_dir / f"feedback_{timestamp}.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return True, str(image_save_path)
        
    except Exception as e:
        return False, str(e)

def run_bulk_test_with_feedback(model_name, model, device, progress_bar, material_selection=None, dataset_folder=None):
    """Run bulk testing with user feedback system and material selection"""
    # Use custom dataset folder if provided, otherwise default to ./dataset/test
    test_dir = Path(dataset_folder) if dataset_folder else Path("./dataset/test")
    if not test_dir.exists():
        st.error(f"❌ Test dataset not found in {test_dir}")
        return None
    
    # Create failure directories
    failure_dirs = create_failure_directories(model_name)
    
    results = []
    total_images = 0
    correct_predictions = 0
    
    # Check if it has subdirectories (material folders)
    has_subdirs = any(d.is_dir() for d in test_dir.iterdir())
    
    if has_subdirs:
        # Material-based structure - process ALL folders
        # Get ground truth labels (Live=1, Fake=0)
        material_labels = {}
        
        for material_dir in test_dir.iterdir():
            if not material_dir.is_dir():
                continue
            
            material_name = material_dir.name
            
            if material_name.lower() == "live":
                material_labels[material_name] = 1  # Live = 1
            else:
                material_labels[material_name] = 0  # Fake = 0
        
        # Process each material folder
        for material_dir in test_dir.iterdir():
            if not material_dir.is_dir():
                continue
            
            material_name = material_dir.name
            true_label = material_labels.get(material_name, 0)
            
            # Check if this folder has subdirectories (Fake materials)
            material_subdirs = [d for d in material_dir.iterdir() if d.is_dir()]
            
            if material_subdirs:
                # Process each material subdirectory (e.g., Ecoflex, Gelatine)
                for subdir in material_subdirs:
                    sub_material_name = subdir.name
                    image_files = list(subdir.glob("*.*"))
                    
                    for img_file in image_files:
                        total_images += 1
                        
                        try:
                            # Load and process image using context manager so file is closed promptly
                            with Image.open(img_file) as image:
                                prediction, confidence = run_single_inference(model, device, image)
                                
                                if prediction is None:
                                    continue
                                
                                # Determine if prediction is correct
                                is_correct = (prediction == true_label)
                                if is_correct:
                                    correct_predictions += 1
                                
                                # Save result with subdirectory name as material
                                result = {
                                    'image': str(img_file),
                                    'material': sub_material_name,
                                    'true_label': true_label,
                                    'predicted_label': prediction,
                                    'confidence': confidence,
                                    'correct': is_correct
                                }
                                results.append(result)

                            # At this point the image file handle has been closed by the context manager
                            # Copy to failure folder if incorrect
                            if not is_correct:
                                failure_path = failure_dirs.get(sub_material_name, failure_dirs.get("Test Images", Path("./failures") / model_name / sub_material_name)) / img_file.name
                                failure_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(img_file, failure_path)
                        except Exception as e:
                            st.error(f"Error processing {img_file}: {e}")
                        
                        # Update progress
                        total_possible = 0
                        for folder in test_dir.iterdir():
                            if folder.is_dir():
                                # Check if folder has subdirectories
                                subdirs = [d for d in folder.iterdir() if d.is_dir()]
                                if subdirs:
                                    # Count images in subdirectories
                                    for subdir in subdirs:
                                        total_possible += len(list(subdir.glob("*.*")))
                                else:
                                    # Count images directly in folder
                                    total_possible += len(list(folder.glob("*.*")))
                        
                        if total_possible > 0:
                            progress = total_images / total_possible
                            progress_bar.progress(progress)
            else:
                # Process direct image files in material folder (like Live)
                image_files = list(material_dir.glob("*.*"))
                
                for img_file in image_files:
                    total_images += 1
                    
                    try:
                        # Load and process image using context manager
                        with Image.open(img_file) as image:
                            prediction, confidence = run_single_inference(model, device, image)
                            
                            if prediction is None:
                                continue
                            
                            # Determine if prediction is correct
                            is_correct = (prediction == true_label)
                            if is_correct:
                                correct_predictions += 1
                            
                            # Save result
                            result = {
                                'image': str(img_file),
                                'material': material_name,
                                'true_label': true_label,
                                'predicted_label': prediction,
                                'confidence': confidence,
                                'correct': is_correct
                            }
                            results.append(result)

                        # Image handle closed here
                        # Copy to failure folder if incorrect
                        if not is_correct:
                            failure_path = failure_dirs.get(material_name, failure_dirs.get("Test Images", Path("./failures") / model_name / material_name)) / img_file.name
                            failure_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(img_file, failure_path)
                    except Exception as e:
                        st.error(f"Error processing {img_file}: {e}")
                    
                    # Update progress
                    total_possible = 0
                    for folder in test_dir.iterdir():
                        if folder.is_dir():
                            # Check if folder has subdirectories
                            subdirs = [d for d in folder.iterdir() if d.is_dir()]
                            if subdirs:
                                # Count images in subdirectories
                                for subdir in subdirs:
                                    total_possible += len(list(subdir.glob("*.*")))
                            else:
                                # Count images directly in folder
                                total_possible += len(list(folder.glob("*.*")))
                    
                    if total_possible > 0:
                        progress = total_images / total_possible
                        progress_bar.progress(progress)
    else:
        # Flat file structure - use material selection
        image_files = list(test_dir.glob("*.*"))
        
        for img_file in image_files:
            total_images += 1
            
            try:
                # Load and process image using context manager
                with Image.open(img_file) as image:
                    prediction, confidence = run_single_inference(model, device, image)
                    
                    if prediction is None:
                        continue
                    
                    # Use material selection if provided
                    material_name = material_selection if material_selection else "Test Images"
                    true_label = 1 if material_selection and "Live" in material_selection else 0  # Assume Live=1, Spoof=0
                    
                    # For flat structure with material selection, we can determine correctness
                    is_correct = (prediction == true_label)
                    if is_correct:
                        correct_predictions += 1
                    
                    # Save result
                    result = {
                        'image': str(img_file),
                        'material': material_name,
                        'true_label': true_label,
                        'predicted_label': prediction,
                        'confidence': confidence,
                        'correct': is_correct
                    }
                    results.append(result)

                # Image closed here
                # Copy to failure folder if incorrect
                if not is_correct:
                    failure_path = failure_dirs.get("Test Images", failure_dirs.get(material_name, Path("./failures") / model_name / "Test Images")) / img_file.name
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_file, failure_path)
            except Exception as e:
                st.error(f"Error processing {img_file}: {e}")
            
            # Update progress
            progress = total_images / len(image_files)
            progress_bar.progress(progress)
    
    return results

def calculate_metrics(results):
    """Calculate comprehensive metrics"""
    if not results:
        return {}
    
    # Overall metrics
    total = len(results)
    
    # For flat structure, we can't calculate accuracy without ground truth
    has_ground_truth = any(r['correct'] != 'unknown' for r in results)
    
    if has_ground_truth:
        correct = sum(1 for r in results if r['correct'] == True)
        overall_accuracy = (correct / total) * 100 if total > 0 else 0
    else:
        correct = 0
        overall_accuracy = 0  # Can't calculate without ground truth
    
    # Material-specific metrics - include all materials even if empty
    material_metrics = {}
    material_results = defaultdict(list)
    
    for result in results:
        material_results[result['material']].append(result)
    
    # Get all possible materials from dataset structure
    dataset_structure = get_test_dataset_structure()
    all_materials = set()
    
    if isinstance(dataset_structure, dict):
        for key, value in dataset_structure.items():
            if isinstance(value, dict):
                # This is Fake folder with material subdirectories
                for sub_material in value.keys():
                    all_materials.add(sub_material)
            else:
                # This is a direct material folder
                all_materials.add(key)
    
    # Add materials from results
    all_materials.update(material_results.keys())
    
    # Calculate metrics for all materials
    for material in sorted(all_materials):
        material_res = material_results.get(material, [])
        material_total = len(material_res)
        
        if has_ground_truth and material_total > 0:
            material_correct = sum(1 for r in material_res if r['correct'] == True)
            material_accuracy = (material_correct / material_total) * 100
            failures = material_total - material_correct
        else:
            material_correct = 0
            material_accuracy = 0 if material_total == 0 else 0  # Can't calculate without ground truth
            failures = sum(1 for r in material_res if r['predicted_label'] == 0)  # Count spoofs as "failures"
        
        material_metrics[material] = {
            'accuracy': material_accuracy,
            'total': material_total,
            'correct': material_correct,
            'failures': failures
        }
    
    # Sort materials by failure rate
    top_offenders = sorted(
        material_metrics.items(),
        key=lambda x: x[1]['failures'],
        reverse=True
    )
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_images': total,
        'correct_predictions': correct,
        'material_metrics': material_metrics,
        'top_offenders': top_offenders,
        'has_ground_truth': has_ground_truth
    }

def plot_confusion_matrix(results):
    """Generate confusion matrix plot"""
    if not results:
        return None
    
    # Check if we have ground truth
    has_ground_truth = any(r['correct'] != 'unknown' for r in results)
    
    if not has_ground_truth:
        # Create a simple prediction distribution plot instead
        pred_labels = [r['predicted_label'] for r in results]
        pred_counts = Counter(pred_labels)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = ['Live (1)', 'Fake (0)']
        counts = [pred_counts.get(1, 0), pred_counts.get(0, 0)]
        
        ax.bar(labels, counts, color=['green', 'red'])
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Distribution (No Ground Truth Available)')
        
        return fig
    
    # Create confusion matrix data
    true_labels = [r['true_label'] for r in results]
    pred_labels = [r['predicted_label'] for r in results]
    
    # Get unique labels (should be 0 and 1)
    labels = sorted(list(set(true_labels + pred_labels)))
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake (0)', 'Live (1)'], 
                yticklabels=['Fake (0)', 'Live (1)'], 
                ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Add summary statistics
    total = len(results)
    correct = sum(1 for r in results if r['correct'] == True)
    accuracy = (correct / total) * 100
    
    # Add text annotation
    ax.text(0.02, 0.98, f'Total: {total}\nCorrect: {correct}\nAccuracy: {accuracy:.1f}%', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'total_inferences': 0,
            'live_detections': 0,
            'spoof_detections': 0,
            'accuracy': 0.0
        }
    if 'user_corrections' not in st.session_state:
        st.session_state.user_corrections = []
    if 'feedback_mode' not in st.session_state:
        st.session_state.feedback_mode = False
    if 'bulk_results' not in st.session_state:
        st.session_state.bulk_results = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    
    # Header
    st.markdown("""
    <div class="main-header">
        🔐 Advanced Fingerprint Spoof Detection with User Feedback
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model selection and feedback mode
    st.sidebar.markdown("### 🔧 Configuration")
    
    # Feedback mode toggle
    feedback_mode = st.sidebar.checkbox(
        "🔄 Enable User Feedback Mode",
        value=st.session_state.get('feedback_mode', False),
        help="Enable to collect user corrections and save misclassified images"
    )
    st.session_state.feedback_mode = feedback_mode
    
    if feedback_mode:
        st.sidebar.success("✅ Feedback mode enabled - corrections will be saved")
    else:
        st.sidebar.info("📝 Feedback mode disabled")
    
    # Model selection
    available_models = get_available_models()
    if not available_models:
        st.error("❌ No models found in ./models/ folder")
        st.stop()  # Use st.stop() instead of return to allow tab navigation
    
    # Create model selection with better labels
    model_options = []
    for model_path in available_models:
        if "Nagaraju_Final_ResNet" in model_path:
            model_options.append(f"🚀 {Path(model_path).name} (Optimized)")
        else:
            model_options.append(f"📊 {Path(model_path).name} (Legacy)")
    
    selected_model_idx = st.sidebar.selectbox(
        "Select Model:",
        range(len(model_options)),
        format_func=lambda x: model_options[x],
        index=0  # First model (optimized if available) is default
    )
    
    selected_model = available_models[selected_model_idx]
    model_name = Path(selected_model).stem
    
    # Show model info
    if "Nagaraju_Final_ResNet" in selected_model:
        st.sidebar.success(f"🚀 Using Optimized Model: {model_name}")
        st.sidebar.info("✨ Enhanced ResNet18 with advanced training and class balancing")
    else:
        st.sidebar.info(f"📊 Using Legacy Model: {model_name}")
    
    # Load model
    try:
        model, device = load_fingerprint_model(selected_model)
        if model is None:
            st.error("❌ Failed to load model")
            st.stop()  # Use st.stop() instead of return to allow tab navigation
        st.sidebar.success(f"✅ Model loaded: {model_name}")
        st.sidebar.info(f"🖥️ Device: {device}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.stop()  # Use st.stop() instead of return to allow tab navigation
    
    # Main tabs
    if SCANNER_AVAILABLE:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🖼️ Single Test", "📊 Bulk Test", "📈 Analysis", "🔄 User Feedback", "🔌 Scanner Input"])
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Single Test", "📊 Bulk Test", "📈 Analysis", "🔄 User Feedback"])
    
    with tab1:
        st.markdown("### 🖼️ Single Image Testing")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload fingerprint image:",
            type=['png', 'jpg', 'jpeg', 'bmp']
        )
        
        if uploaded_file is not None:
                # Load image using context manager
                with Image.open(uploaded_file) as image:
                    # Run inference
                    prediction, confidence = run_single_inference(model, device, image)
                    
                    # Display results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.image(image, caption="📷 Original Image", width='content')
                    # Show preprocessed version
                    preprocessed_tensor = preprocess_image(image)
                    if preprocessed_tensor is not None:
                        # Convert tensor back to image for display
                        preprocessed_np = preprocessed_tensor.permute(1, 2, 0).numpy()
                        # Denormalize for display
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        preprocessed_np = preprocessed_np * std + mean
                        preprocessed_np = np.clip(preprocessed_np, 0, 1)
                        st.image(preprocessed_np, caption="🔧 Preprocessed Image", width='content')
                
                if prediction is not None:
                    with col2:
                        # Display confidence scores for both classes
                        st.markdown("### 📊 Prediction Results")
                        
                        # Get raw probabilities
                        image_tensor = preprocess_image(image)
                        if image_tensor is not None:
                            image_tensor = image_tensor.unsqueeze(0).to(device)
                            with torch.no_grad():
                                output = model(image_tensor)
                                probabilities = torch.softmax(output, dim=1)
                                probs = probabilities[0].cpu().numpy()
                            
                            st.write("**Class Probabilities:**")
                            st.write(f"🟢 Live (Class 1): {probs[1]:.3f} ({probs[1]*100:.1f}%)")
                            st.write(f"🔴 Spoof (Class 0): {probs[0]:.3f} ({probs[0]*100:.1f}%)")
                        
                        # Final prediction
                        if prediction == 1:
                            st.markdown("""
                            <div class="success-box">
                                🔍 RESULT: LIVE FINGERPRINT
                            </div>
                            """, unsafe_allow_html=True)
                            model_result = "Live"
                        else:
                            st.markdown("""
                            <div class="danger-box">
                                ⚠️ RESULT: SPOOF DETECTED
                            </div>
                            """, unsafe_allow_html=True)
                            model_result = "Spoof"
                    
                    st.markdown(f"""
                    <div class="info-box">
                        🎯 Confidence: {confidence:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # User feedback section
                    if st.session_state.feedback_mode:
                        st.markdown("### 🔄 User Correction")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            user_label = st.radio(
                                "Is this prediction correct?",
                                ["Live", "Spoof"],
                                key=f"feedback_{uploaded_file.name}"
                            )
                            
                            # Material selection dropdown
                            material_types = ["Silicone", "Gelatin", "Play-Doh", "Ecoflex", "Latex", "Body Double", "Other"]
                            material_type = st.selectbox(
                                "Material Type (if fake):",
                                material_types,
                                key=f"material_{uploaded_file.name}",
                                index=0
                            )
                        
                        with col2:
                            if st.button("Submit Feedback", key=f"submit_{uploaded_file.name}"):
                                feedback_saved, result = save_user_feedback(
                                    image,
                                    uploaded_file.name,
                                    prediction,
                                    1 if user_label == "Live" else 0,
                                    confidence,
                                    material_type if user_label == "Spoof" else None
                                )
                                
                                if feedback_saved:
                                    st.success(f"✅ Feedback saved! Image saved to: {result}")
                                    # Add to corrections list
                                    st.session_state.user_corrections.append({
                                        'image': uploaded_file.name,
                                        'model_prediction': model_result,
                                        'user_label': user_label,
                                        'confidence': confidence,
                                        'material_type': material_type if user_label == "Spoof" else None,
                                        'saved_image_path': result,
                                        'timestamp': datetime.now().isoformat()
                                    })
                                else:
                                    st.error(f"❌ Failed to save feedback: {result}")
    
    with tab2:
        st.markdown("### 📊 Bulk Dataset Testing")
        
        # Dataset folder selection
        st.markdown("#### 📁 Dataset Folder Selection")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Folder selection options
            dataset_folder = st.text_input(
                "Enter Dataset Folder Path:",
                value="./dataset/test",
                key="dataset_folder_input",
                help="Enter the path to your test dataset folder (e.g., ./dataset/test or C:/data/fingerprint_test)"
            )
            
            # Quick access buttons for common folders
            st.markdown("**Quick Access:**")
            quick_col1, quick_col2, quick_col3 = st.columns(3)
            
            with quick_col1:
                if st.button("📂 Default Test", key="default_test"):
                    # Clear the session state and rerun to update the text input
                    if 'dataset_folder_input' in st.session_state:
                        del st.session_state['dataset_folder_input']
                    st.rerun()
            
            with quick_col2:
                if st.button("📂 Browse Files", key="browse_files"):
                    st.info("📁 **How to browse:**\n1. Open File Explorer\n2. Navigate to your dataset folder\n3. Click in the address bar\n4. Copy the path and paste it above")
            
            with quick_col3:
                if st.button("📂 Examples", key="example_paths"):
                    st.info("💡 **Example paths:**\n• `./dataset/test`\n• `C:/Users/YourName/Desktop/fingerprint_data`\n• `D:/Research/biometric_test`\n• `../other_project/test_data`")
        
        with col2:
            # Show current selection info
            current_folder = dataset_folder
            test_path = Path(current_folder)
            
            if test_path.exists():
                st.success(f"✅ Folder Found")
                st.code(f"{current_folder}")
                
                # Count files and folders
                if test_path.is_dir():
                    total_files = len(list(test_path.rglob("*.*")))
                    subdirs = len([d for d in test_path.iterdir() if d.is_dir()])
                    st.write(f"📊 {total_files} files")
                    st.write(f"📁 {subdirs} subfolders")
                    
                    # Show folder structure preview
                    if subdirs > 0:
                        st.write("**Structure:**")
                        for item in list(test_path.iterdir())[:5]:  # Show first 5 items
                            if item.is_dir():
                                st.write(f"📁 {item.name}/")
                            else:
                                st.write(f"📄 {item.name}")
                        if subdirs > 5:
                            st.write(f"... and {subdirs - 5} more")
            else:
                st.error(f"❌ Folder Not Found")
                st.code(f"{current_folder}")
                st.stop()  # Use st.stop() instead of return to allow tab navigation
        
        # Update the dataset structure check to use selected folder
        def get_custom_dataset_structure(folder_path):
            """Analyze custom dataset structure"""
            test_dir = Path(folder_path)
            if not test_dir.exists():
                return {}
            
            structure = {}
            
            # Check if it has subdirectories (material folders)
            has_subdirs = any(d.is_dir() for d in test_dir.iterdir())
            
            if has_subdirs:
                # Material-based structure - check subdirectories
                for material_dir in test_dir.iterdir():
                    if material_dir.is_dir():
                        material_name = material_dir.name
                        
                        # Check if this material folder has subdirectories
                        material_subdirs = [d for d in material_dir.iterdir() if d.is_dir()]
                        
                        if material_subdirs:
                            # Has material type subdirectories
                            structure[material_name] = {}
                            for subdir in material_subdirs:
                                image_files = list(subdir.glob("*.*"))
                                structure[material_name][subdir.name] = len(image_files)
                        else:
                            # Direct image files in material folder
                            image_files = list(material_dir.glob("*.*"))
                            structure[material_name] = len(image_files)
            else:
                # Flat file structure - treat all files as one category
                image_files = list(test_dir.glob("*.*"))
                structure["Test Images"] = len(image_files)
            
            return structure
        
        # Check dataset structure using selected folder
        dataset_structure = get_custom_dataset_structure(dataset_folder)
        if not dataset_structure:
            st.error(f"❌ Dataset not found in {dataset_folder}")
            st.stop()  # Use st.stop() instead of return to allow tab navigation
        
        # Check if dataset has subdirectories (using selected folder)
        test_dir = Path(dataset_folder)
        has_subdirs = any(d.is_dir() for d in test_dir.iterdir())
        
        if not has_subdirs:
            # Show material selection for flat structure
            st.markdown("#### 🎯 Material Selection")
            st.info("📁 Your dataset has a flat structure. Select the material type for all images:")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                material_options = [
                    "Live Fingerprint",
                    "Silicone",
                    "Gelatin", 
                    "Play-Doh",
                    "Ecoflex",
                    "Latex",
                    "Body Double",
                    "Unknown Material"
                ]
                
                selected_material = st.selectbox(
                    "Select Material Type:",
                    material_options,
                    key="bulk_material_selection",
                    help="This will be used as the ground truth for all images in the test folder"
                )
            
            with col2:
                if selected_material:
                    if "Live" in selected_material:
                        st.markdown("""
                        <div class="success-box">
                            🟢 LIVE
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="danger-box">
                            🔴 FAKE
                        </div>
                        """, unsafe_allow_html=True)
        
        st.write("**Dataset Structure:**")
        def display_structure(structure, indent=0):
            for key, value in structure.items():
                if isinstance(value, dict):
                    st.write("  " * indent + f"📁 {key}/")
                    display_structure(value, indent + 1)
                else:
                    st.write("  " * indent + f"📁 {key}/: {value} images")
        
        display_structure(dataset_structure)
        
        # Run bulk test button
        if st.button("🚀 Run Full Dataset Test", type="primary"):
            st.info("🔄 Running bulk test... This may take a while.")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run bulk test with material selection and custom dataset folder
            material_for_test = selected_material if not has_subdirs else None
            results = run_bulk_test_with_feedback(model_name, model, device, progress_bar, material_for_test, dataset_folder)
            
            if results:
                # Calculate metrics
                metrics = calculate_metrics(results)
                
                # Store results in session state
                st.session_state.bulk_results = results
                st.session_state.metrics = metrics
                
                if metrics['has_ground_truth']:
                    st.success(f"✅ Bulk test completed! Overall accuracy: {metrics['overall_accuracy']:.1f}%")
                else:
                    st.success(f"✅ Bulk test completed! Processed {metrics['total_images']} images")
            else:
                st.error("❌ Bulk test failed!")
    
    with tab3:
        st.markdown("### 📈 Analysis Results")
        
        # Check if results exist
        if st.session_state.get('bulk_results') is not None:
            results = st.session_state.bulk_results
            metrics = st.session_state.metrics
            
            # Overall metrics
            st.markdown("#### 📊 Overall Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if metrics['has_ground_truth']:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{metrics['overall_accuracy']:.1f}%</h3>
                        <p>Overall Accuracy</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h3>N/A</h3>
                        <p>No Ground Truth Available</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{metrics['total_images']}</h3>
                    <p>Total Images Tested</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if metrics['has_ground_truth']:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{metrics['correct_predictions']}</h3>
                        <p>Correct Predictions</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>-</h3>
                        <p>Cannot Calculate</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Material-specific metrics
            st.markdown("#### 🎯 Material-Specific Performance")
            
            material_data = []
            for material, metric in metrics['material_metrics'].items():
                status = "🟢" if metric['total'] > 0 else "🔴"
                material_data.append({
                    'Material': f"{status} {material}",
                    'Accuracy': f"{metric['accuracy']:.1f}%" if metric['total'] > 0 else "N/A",
                    'Total': metric['total'],
                    'Failures': metric['failures'],
                    'Status': "Has Data" if metric['total'] > 0 else "Empty Folder"
                })
            
            df_materials = pd.DataFrame(material_data)
            st.dataframe(df_materials, width='stretch')
            
            # Show empty folder warning based on dataset structure, not processed results
            dataset_structure = get_test_dataset_structure()
            empty_materials = []
            
            if isinstance(dataset_structure, dict):
                for key, value in dataset_structure.items():
                    if isinstance(value, dict):
                        # This is Fake folder with material subdirectories
                        for sub_material, count in value.items():
                            if count == 0:
                                empty_materials.append(sub_material)
                    else:
                        # This is a direct material folder
                        if value == 0:
                            empty_materials.append(key)
            
            if empty_materials:
                st.warning(f"⚠️ Empty folders detected: {', '.join(empty_materials)}. Add images to these folders to test fake materials.")
            else:
                st.success("✅ All folders have images! Ready for comprehensive testing.")
            
            # Show dataset structure summary
            st.markdown("#### 📁 Dataset Structure Summary")
            dataset_structure = get_test_dataset_structure()
            def display_structure_summary(structure, indent=0):
                for key, value in structure.items():
                    if isinstance(value, dict):
                        st.write("  " * indent + f"📁 {key}/")
                        display_structure_summary(value, indent + 1)
                    else:
                        status = "✅" if value > 0 else "❌"
                        st.write("  " * indent + f"{status} 📁 {key}/: {value} images")
            
            display_structure_summary(dataset_structure)
            
            # Top offenders
            st.markdown("#### ⚠️ Top Offenders (Most Failed Materials)")
            
            # Filter out materials with 0 total images to avoid division by zero
            valid_offenders = [(material, metric) for material, metric in metrics['top_offenders'] if metric['total'] > 0]
            
            if valid_offenders:
                for i, (material, metric) in enumerate(valid_offenders[:5]):
                    failure_rate = (metric['failures'] / metric['total']) * 100
                    st.write(f"{i+1}. **{material}**: {metric['failures']}/{metric['total']} failures ({failure_rate:.1f}%)")
            else:
                st.info("📊 No materials with failures detected or all folders are empty.")
            
            # Confusion matrix
            st.markdown("#### 📈 Confusion Matrix")
            
            fig = plot_confusion_matrix(st.session_state.bulk_results)
            if fig:
                st.pyplot(fig)
            
            # Download results
            st.markdown("#### 💾 Export Results")
            
            # Create CSV of results
            df_results = pd.DataFrame(st.session_state.bulk_results)
            csv = df_results.to_csv(index=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"test_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create JSON of metrics
                metrics_json = json.dumps(metrics, indent=2)
                st.download_button(
                    label="📥 Download Metrics JSON",
                    data=metrics_json,
                    file_name=f"metrics_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("Please run a Bulk Test in Tab 2 to see analysis.")
    
    with tab4:
        st.markdown("### 🔄 User Feedback History")
        
        if st.session_state.feedback_mode:
            # Show feedback directory structure
            feedback_dir = Path("./user_feedback")
            if feedback_dir.exists():
                st.markdown("#### 📁 Feedback Directory Structure")
                
                # Display directory tree
                def display_directory_tree(path, indent=0):
                    if path.is_dir():
                        st.write("  " * indent + f"📁 {path.name}/")
                        for item in sorted(path.iterdir()):
                            display_directory_tree(item, indent + 1)
                    else:
                        st.write("  " * indent + f"📄 {path.name}")
                
                display_directory_tree(feedback_dir)
                
                # Show statistics
                live_images = len(list(feedback_dir.glob("Live/**/*.png")))
                spoof_images = len(list(feedback_dir.glob("Spoof/**/*.png")))
                
                st.markdown("#### 📊 Image Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{live_images}</h3>
                        <p>Live Images</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{spoof_images}</h3>
                        <p>Spoof Images</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_feedback_images = live_images + spoof_images
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{total_feedback_images}</h3>
                        <p>Total Images</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.session_state.user_corrections:
                st.markdown("#### 📝 User Corrections")
                
                # Display corrections in a table
                corrections_df = pd.DataFrame(st.session_state.user_corrections)
                st.dataframe(corrections_df, width='stretch')
                
                # Summary statistics
                total_corrections = len(st.session_state.user_corrections)
                model_wrong = sum(1 for c in st.session_state.user_corrections if c['model_prediction'] != c['user_label'])
                
                st.markdown("#### 📊 Feedback Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{total_corrections}</h3>
                        <p>Total Corrections</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{model_wrong}</h3>
                        <p>Model Was Wrong</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    accuracy_rate = ((total_corrections - model_wrong) / total_corrections * 100) if total_corrections > 0 else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{accuracy_rate:.1f}%</h3>
                        <p>User Agreement Rate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Material-specific statistics
                material_stats = defaultdict(int)
                for correction in st.session_state.user_corrections:
                    if correction['material_type']:
                        material_stats[correction['material_type']] += 1
                
                if material_stats:
                    st.markdown("#### 🎯 Material-Specific Corrections")
                    material_data = []
                    for material, count in material_stats.items():
                        material_data.append({
                            'Material': material,
                            'Corrections': count
                        })
                    
                    material_df = pd.DataFrame(material_data)
                    st.dataframe(material_df, width='stretch')
                
                # Download feedback data
                if st.button("📥 Download Feedback Data"):
                    feedback_df = pd.DataFrame(st.session_state.user_corrections)
                    feedback_csv = feedback_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Feedback CSV",
                        data=feedback_csv,
                        file_name=f"user_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("📝 No user feedback yet. Enable feedback mode and start testing!")
        else:
            st.info("🔄 Enable feedback mode to see user correction history")
    
    # Scanner Input Tab
    with tab5:
        st.markdown("### 🔌 Live Scanner Input")
        
        # Always show scanner options
        st.markdown("#### 📋 Scanner Setup")
        
        # Scanner availability check
        scanner_available = SCANNER_AVAILABLE
        
        if scanner_available:
            try:
                from fingerprint_scanner import FingerprintScanner
                scanner = FingerprintScanner()
                st.success("✅ Scanner module loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading scanner: {e}")
                scanner_available = False
        else:
            st.warning("⚠️ Scanner module not available")
        
        # Scanner connection options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Detect Scanners", key="detect_scanners"):
                if scanner_available:
                    try:
                        devices = scanner.detect_scanners()
                        if devices:
                            st.success(f"✅ Found {len(devices)} scanner(s)")
                            st.session_state.scanner_devices = devices
                        else:
                            st.warning("⚠️ No scanners detected")
                            st.session_state.scanner_devices = []
                    except Exception as e:
                        st.error(f"❌ Error detecting scanners: {e}")
                else:
                    st.error("❌ Scanner module not available")
        
        with col2:
            if st.button("📷 Test USB Camera", key="test_usb_camera"):
                try:
                    # Test with OpenCV to see if any cameras are available
                    test_cap = cv2.VideoCapture(0)
                    if test_cap.isOpened():
                        ret, frame = test_cap.read()
                        if ret:
                            st.success("✅ USB Camera detected and working!")
                            st.session_state.usb_camera_available = True
                        else:
                            st.error("❌ USB Camera found but no frame captured")
                        test_cap.release()
                    else:
                        st.error("❌ No USB Camera found")
                        st.session_state.usb_camera_available = False
                except Exception as e:
                    st.error(f"❌ Error testing USB Camera: {e}")
        
        with col3:
            if st.button("📋 Show All Options", key="show_options"):
                st.info("📋 Scanner Options:")
                st.write("1. USB Camera (most common)")
                st.write("2. DigitalPersona Scanner")
                st.write("3. SecuGen Scanner")
                st.write("4. Generic Webcam")
        
        # Scanner selection and controls
        st.markdown("#### 🔌 Scanner Connection")
        
        # Input method selection
        input_method = st.radio(
            "Select Input Method:",
            ["USB Camera", "Upload Image", "Test Mode"],
            key="input_method_selection"
        )
        
        if input_method == "USB Camera":
            st.markdown("##### 📷 USB Camera Setup")
            
            # Camera index selection
            camera_idx = st.selectbox(
                "Select Camera Index:",
                [0, 1, 2, 3, 4],
                index=0,
                key="camera_index_selection"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔌 Connect Camera", key="connect_camera"):
                    try:
                        cap = cv2.VideoCapture(camera_idx)
                        if cap.isOpened():
                            st.success(f"✅ Camera {camera_idx} connected!")
                            st.session_state.camera_connected = True
                            st.session_state.current_camera = cap
                        else:
                            st.error(f"❌ Failed to connect to Camera {camera_idx}")
                    except Exception as e:
                        st.error(f"❌ Error connecting camera: {e}")
            
            with col2:
                if st.button("📹 Start Preview", key="start_preview"):
                    if st.session_state.get('camera_connected', False):
                        st.session_state.preview_active = True
                        st.success("✅ Preview started")
                    else:
                        st.error("❌ Please connect camera first")
            
            with col3:
                if st.button("⏹️ Stop Preview", key="stop_preview"):
                    st.session_state.preview_active = False
                    if 'current_camera' in st.session_state:
                        st.session_state.current_camera.release()
                    st.info("⏹️ Preview stopped")
            
            # Camera preview
            if st.session_state.get('preview_active', False):
                st.markdown("##### 📹 Live Preview")
                
                try:
                    if 'current_camera' in st.session_state:
                        cap = st.session_state.current_camera
                        ret, frame = cap.read()
                        if ret:
                            # Convert frame for display
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(rgb_frame, channels="RGB", width=400, caption="Live Camera Feed")
                            
                            # Capture button
                            if st.button("📸 Capture Frame", key="capture_from_camera"):
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                save_path = Path(f"./camera_capture_{timestamp}.png")
                                
                                # Save frame
                                cv2.imwrite(str(save_path), frame)
                                
                                # Convert to PIL for analysis
                                pil_image = Image.fromarray(rgb_frame)
                                st.session_state.captured_image = pil_image
                                st.session_state.captured_path = str(save_path)
                                st.success(f"✅ Frame captured: {save_path.name}")
                        else:
                            st.error("❌ Failed to capture frame")
                except Exception as e:
                    st.error(f"❌ Error in preview: {e}")
                
                # Auto-refresh
                time.sleep(0.1)
                st.rerun()
        
        elif input_method == "Upload Image":
            st.markdown("##### 📁 Upload Image")
            
            uploaded_file = st.file_uploader(
                "Upload fingerprint image:",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="scanner_upload"
            )
            
            if uploaded_file is not None:
                with Image.open(uploaded_file) as image:
                    st.session_state.captured_image = image
                    st.session_state.captured_path = uploaded_file.name
                    st.success("✅ Image uploaded successfully!")
                    st.image(image, width=400, caption="Uploaded Image")
        
        elif input_method == "Test Mode":
            st.markdown("##### 🧪 Test Mode")
            
            # Load a sample image from dataset
            test_dir = Path("./dataset/test")
            if test_dir.exists():
                # Find a sample image
                sample_images = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    sample_images.extend(test_dir.rglob(ext))
                
                if sample_images:
                    sample_path = sample_images[0]
                    with Image.open(sample_path) as sample_image:
                        st.session_state.captured_image = sample_image
                        st.session_state.captured_path = str(sample_path)
                        st.success("✅ Test image loaded!")
                        st.image(sample_image, width=400, caption="Test Image")
                else:
                    st.warning("⚠️ No test images found in dataset")
            else:
                st.warning("⚠️ Dataset directory not found")
        
        # Show captured image and analysis
        if 'captured_image' in st.session_state:
            st.markdown("#### 📸 Captured Image Analysis")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(st.session_state.captured_image, width=400, caption="Captured Image")
                
                # Analysis button
                if st.button("🔄 Analyze Image", key="analyze_captured_image"):
                    try:
                        image = st.session_state.captured_image
                        prediction, confidence = run_single_inference(model, device, image)
                        
                        if prediction is not None:
                            # Store analysis results
                            st.session_state.scanner_analysis = {
                                'image': st.session_state.captured_image,
                                'prediction': prediction,
                                'confidence': confidence,
                                'timestamp': datetime.now().isoformat()
                            }
                            st.success("✅ Image analyzed!")
                    except Exception as e:
                        st.error(f"❌ Error analyzing image: {e}")
            
            with col2:
                if 'scanner_analysis' in st.session_state:
                    analysis = st.session_state.scanner_analysis
                    image = analysis['image']
                    prediction = analysis['prediction']
                    confidence = analysis['confidence']
                    
                    # Display results
                    st.markdown("### 📊 Analysis Results")
                    
                    # Get raw probabilities
                    image_tensor = preprocess_image(image)
                    if image_tensor is not None:
                        image_tensor = image_tensor.unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = model(image_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            probs = probabilities[0].cpu().numpy()
                        
                        st.write("**Class Probabilities:**")
                        st.write(f"🟢 Live (Class 1): {probs[1]:.3f} ({probs[1]*100:.1f}%)")
                        st.write(f"🔴 Spoof (Class 0): {probs[0]:.3f} ({probs[0]*100:.1f}%)")
                    
                    # Final prediction
                    if prediction == 1:
                        st.markdown("""
                        <div class="success-box">
                            🔍 RESULT: LIVE FINGERPRINT
                        </div>
                        """, unsafe_allow_html=True)
                        model_result = "Live"
                    else:
                        st.markdown("""
                        <div class="danger-box">
                            ⚠️ RESULT: SPOOF DETECTED
                        </div>
                        """, unsafe_allow_html=True)
                        model_result = "Spoof"
                    
                    st.markdown(f"""
                    <div class="info-box">
                        🎯 Confidence: {confidence:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # User feedback
                    if st.session_state.feedback_mode:
                        st.markdown("### 🔄 User Feedback")
                        
                        user_label = st.radio(
                            "Is this prediction correct?",
                            ["Live", "Spoof"],
                            key=f"scanner_feedback_{analysis['timestamp']}"
                        )
                        
                        material_types = ["Silicone", "Gelatin", "Play-Doh", "Ecoflex", "Latex", "Body Double", "Other"]
                        material_type = st.selectbox(
                            "Material Type (if fake):",
                            material_types,
                            key=f"scanner_material_{analysis['timestamp']}",
                            index=0
                        )
                        
                        if st.button("Submit Feedback", key=f"scanner_submit_{analysis['timestamp']}"):
                            feedback_saved, result = save_user_feedback(
                                image,
                                f"scanner_capture_{analysis['timestamp']}.png",
                                prediction,
                                1 if user_label == "Live" else 0,
                                confidence,
                                material_type if user_label == "Spoof" else None
                            )
                            
                            if feedback_saved:
                                st.success(f"✅ Feedback saved!")
                                st.session_state.user_corrections.append({
                                    'image': f"scanner_capture_{analysis['timestamp']}.png",
                                    'source': 'Scanner',
                                    'model_prediction': model_result,
                                    'user_label': user_label,
                                    'confidence': confidence,
                                    'material_type': material_type if user_label == "Spoof" else None,
                                    'saved_image_path': result,
                                    'timestamp': analysis['timestamp']
                                })
                            else:
                                st.error(f"❌ Failed to save feedback: {result}")
            
            # Clear button
            if st.button("🗑️ Clear Captured Image", key="clear_captured_image"):
                if 'captured_image' in st.session_state:
                    del st.session_state.captured_image
                if 'captured_path' in st.session_state:
                    del st.session_state.captured_path
                if 'scanner_analysis' in st.session_state:
                    del st.session_state.scanner_analysis
                st.info("🗑️ Captured data cleared")

if __name__ == "__main__":
    main()
