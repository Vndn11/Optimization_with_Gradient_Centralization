# **🖼️ CIFAR-100 Image Classification with Optimized Deep Learning Training**

## **📌 Overview**
This project focuses on **image classification using the CIFAR-100 dataset**. The primary objective is to **train deep learning models with various optimization techniques**, including **Stochastic Gradient Descent (SGD), Adam, RMSprop, and Gradient Centralization (GC) variants**, to improve training efficiency and classification accuracy.

By implementing **different optimization strategies**, we analyze how various gradient-based methods influence model performance and convergence. The project also explores **training time, loss curves, and accuracy trends** to determine the most effective approach for CIFAR-100 classification.

---

## **📂 Dataset**
The **CIFAR-100** dataset consists of:
- **60,000** images in **100 classes**, each containing **600 images**.
- **50,000** training images and **10,000** test images.
- Each image is **32x32 pixels**, with **3 color channels (RGB)**.
- **Hierarchical labeling** – Each image has:
  - **Fine-grained labels** (specific category, e.g., **"oak tree"**).
  - **Coarse labels** (broader superclass, e.g., **"trees"**).

### **🔹 Example Classes in CIFAR-100**
- Fine Labels: **apple, dolphin, skyscraper, television, tractor, oak tree, etc.**
- Coarse Labels: **fish, fruit & vegetables, trees, vehicles, etc.**

---

## **🛠 Tech Stack**
| Technology       | Purpose |
|-----------------|------------------------------|
| **Python**      | Core programming language |
| **NumPy**       | Numerical computations |
| **TensorFlow/Keras** | Deep Learning framework |
| **Matplotlib**  | Data visualization |
| **Pickle**      | Data serialization |
| **Pandas**      | Data handling |
| **Time Module** | Training time analysis |

---

## **📌 Project Workflow**
### **1️⃣ Data Preprocessing**
- **Loading dataset** using `pickle`.
- **Normalizing image pixel values** to **[0,1]** for faster convergence.
- **One-hot encoding labels** for multi-class classification.
- **Splitting dataset** into **training and testing sets**.

---

### **2️⃣ Model Architecture**
The model consists of a **fully connected neural network** (MLP-style architecture), using:
- **Input layer**: 3072 features (**32x32x3** pixels).
- **Hidden layers**: Multiple **dense layers with ReLU activation**.
- **Output layer**: Softmax activation for **100-category classification**.

#### **🔹 Model Summary**
```python
model = Sequential([
    Dense(512, activation='relu', input_shape=(3072,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(100, activation='softmax')
])
```

---

### **3️⃣ Optimizers & Training Strategies**
The project implements multiple **gradient-based optimization algorithms**:

| Optimizer  | Description |
|------------|--------------------------------|
| **SGD**  | Basic Stochastic Gradient Descent |
| **SGDM**  | SGD with Momentum for stability |
| **RMSprop**  | Root Mean Square Propagation |
| **Adam**  | Adaptive Moment Estimation |
| **Gradient Centralization (GC)** | Enhances gradient updates for better training stability |

Each optimizer **adjusts network weights differently**, impacting **convergence speed, loss reduction, and accuracy**.

#### **🔹 Training Process**
- **Loss Function**: **Categorical Cross-Entropy**  
- **Optimization**: Various optimizers tested  
- **Evaluation Metrics**: Accuracy, Loss, Training Time  
- **Backpropagation**: Updates model weights  
- **Learning Rate Adjustment**: Ensures stable learning  

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=30)
```

---

### **4️⃣ Model Training & Performance Analysis**
- The model is trained using **multiple optimizers**.
- **Performance metrics** are recorded:
  - **Training time per epoch**
  - **Loss values over epochs**
  - **Accuracy trends**
- **Final accuracy and loss values** are stored in results files.

---

### **5️⃣ Testing & Evaluation**
- The trained model is evaluated on the **CIFAR-100 test set**.
- **Predictions are generated** using forward propagation.
- **Comparison of optimizers** in terms of:
  - Final accuracy
  - Training time
  - Loss minimization

```python
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

---

## **📊 Results & Insights**
### **🔹 Performance Comparison of Optimizers**
| Optimizer  | Training Time | Test Accuracy |
|------------|--------------|--------------|
| **SGD** | Slow | Lower accuracy |
| **SGDM** | Medium | Stable learning |
| **RMSprop** | Fast | Good accuracy |
| **Adam** | Very Fast | High accuracy |
| **GC-SGD** | Medium | More stable than SGD |

### **🔹 Key Takeaways**
✅ **Gradient Centralization (GC)** helps stabilize learning.  
✅ **Adam and RMSprop** provide **better convergence** than SGD.  
✅ **Momentum-based optimizers** smoothen the training curve.  

---

## **📈 Visualizations**
- **Loss vs. Epochs Curve**
- **Accuracy vs. Epochs Trend**
- **Training Time Comparison**

```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Training Performance')
plt.show()
```

---

## **📂 Repository Structure**
```
📂 CIFAR-100-Classification/
│── 📄 Analysis.py             # Data analysis and visualization
│── 📄 final_ml_project.py     # Model training script
│── 📄 Testing_model.py        # Model testing script
│── 📄 README.md               # Project documentation
│── 📄 requirements.txt        # Required dependencies
```

---

## **🚀 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/CIFAR-100-Classification.git
cd CIFAR-100-Classification
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Training**
```bash
python final_ml_project.py
```

### **4️⃣ Test the Model**
```bash
python Testing_model.py
```

---

## **📌 Future Enhancements**
✅ Implement **Convolutional Neural Networks (CNNs)** for better feature extraction.  
✅ Introduce **Data Augmentation** for improved generalization.  
✅ Optimize hyperparameters using **Grid Search & Bayesian Optimization**.  
✅ Deploy as a **web application** for real-time image classification.  
