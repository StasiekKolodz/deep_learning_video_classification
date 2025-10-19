# Model Results - Task 4.1 (No Data Leakage)

## Per-Frame Model

### Training Results
- **Best Training Accuracy:** 98.60%
- **Best Validation Accuracy:** 41.67%
- **Epochs Trained:** 15

### Test Results
- **Test Accuracy (Top-1):** 45.83%
- **Test Accuracy (Top-5):** 90.00%

---

## Late Fusion Model

### Training Results
- **Best Training Accuracy:** 100.00%
- **Best Validation Accuracy:** 53.33%
- **Epochs Trained:** 29

### Test Results
- **Test Accuracy (Top-1):** 47.50%
- **Test Accuracy (Top-5):** 85.83%

---

## 3D CNN Model

### Training Results
- **Best Training Accuracy:** 100.00%
- **Best Validation Accuracy:** 37.50%
- **Epochs Trained:** 86

### Test Results
- **Test Accuracy (Top-1):** 25.00%
- **Test Accuracy (Top-5):** 75.83%

---

## Early Fusion Model

### Training Results
- **Best Training Accuracy:** 100.00%
- **Best Validation Accuracy:** 32.50%
- **Epochs Trained:** 84

### Test Results
- **Test Accuracy (Top-1):** 30.00%
- **Test Accuracy (Top-5):** 68.33%

---

## Summary - All Models

| Model      | Best Train Acc | Best Val Acc | Test Acc (Top-1) | Test Acc (Top-5) |
|------------|----------------|--------------|------------------|------------------|
| Late       | 100.00%        | 53.33%       | **47.50%**       | 85.83%           |
| Per-Frame  | 98.60%         | 41.67%       | 45.83%           | 90.00%           |
| Early      | 100.00%        | 32.50%       | 30.00%           | 68.33%           |
| 3D CNN     | 100.00%        | 37.50%       | 25.00%           | 75.83%           |

### Best Model

**Late Fusion** achieved the highest test accuracy: **47.50%**

