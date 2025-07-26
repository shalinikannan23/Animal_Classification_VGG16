# Animal Classifier using VGG16 (PyTorch)

This project is a **deep learning model built with PyTorch** to classify images of **Cats, Dogs, and Pandas** using **transfer learning with VGG16**.

---

## Project Overview
- Uses **pretrained VGG16** on ImageNet as a feature extractor.
- Only the final classification layer is trained (transfer learning).
- Dataset: Custom dataset with folders:
  ```
  archive/animals/
      cats/
      dogs/
      panda/
  ```
- Model outputs one of three classes: `cat`, `dog`, `panda`.

---

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/animal-classifier-vgg16.git
cd animal-classifier-vgg16
```

### 2. Install Dependencies
```bash
pip install torch torchvision matplotlib
```

### 3. Dataset
Ensure your dataset is organized as:
```
archive/animals/cats
archive/animals/dogs
archive/animals/panda
```

---

## Model Architecture

**Base Model:** VGG16 pretrained on ImageNet  
**Classifier Modified:**
- Original `nn.Linear(4096, 1000)` → replaced with `nn.Linear(4096, 3)` for 3 classes.

**Training Strategy:**
- Freeze all convolutional layers.
- Train only the final classifier layer.

---

## Code Workflow

### **1. Data Loading**
```python
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
dataset = datasets.ImageFolder("archive/animals", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### **2. Model Preparation**
```python
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(4096, 3)
```

### **3. Loss & Optimizer**
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
```

### **4. Training Loop**
```python
for epoch in range(10):
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Calculations

### **1. Number of Trainable Parameters**

- VGG16 total parameters: **≈138 million**
- After freezing convolutional layers, only the classifier parameters are trained.
- Final layer parameters:  
  `4096 x 3 + 3 = 12,291` trainable parameters.

This makes training **very fast** because 99% of parameters are frozen.

---

### **2. FLOPs (Operations)**
- VGG16 (full): ~15.5 GFLOPs per 224x224 image.
- Since you freeze layers, compute cost is the same for forward pass, but **only the classifier's gradients** are computed.

---

### **3. Output Shape Calculation**

Input: `(32, 3, 224, 224)`  
Output after classifier: `(32, 3)`  

Softmax probabilities correspond to `[cat, dog, panda]`.

---
## Output

<img width="770" height="388" alt="image" src="https://github.com/user-attachments/assets/a373ffbb-4bd3-4739-a812-623e4d45223c" />
<img width="789" height="699" alt="image" src="https://github.com/user-attachments/assets/b929346c-1d78-4d9f-9aef-45ea24d75747" />
<img width="611" height="264" alt="image" src="https://github.com/user-attachments/assets/9fe95cd0-9901-4351-ace0-05fced4a0c50" />
<img width="789" height="699" alt="image" src="https://github.com/user-attachments/assets/292e9daf-8dd3-45e1-8ce3-9ac0081caa07" />
<img width="1990" height="313" alt="image" src="https://github.com/user-attachments/assets/11fcdd8a-b617-433b-83c8-1b7d6958275f" />


## Results

- **Input size:** 224 × 224
- **Optimizer:** Adam
- **Loss:** CrossEntropyLoss
- **Output classes:** 3
- **Expected accuracy:** 90%+ with sufficient data

---

## Future Work
- Add validation/test data split.
- Use learning rate scheduler.
- Experiment with **VGG19** or **ResNet50**.

---
