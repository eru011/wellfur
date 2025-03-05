# WellFur - Backend (Machine Learning Model)

## Overview
WellFur's backend is in its early stages of development and serves as the core machine learning engine for detecting pet skin diseases. This system runs entirely offline and utilizes a MobileNetV3-based Convolutional Neural Network (CNN) model to process images and return predictions. The goal is to refine accuracy over time and optimize performance for mobile deployment.

## Features
- **Offline Processing**: Runs locally without requiring an internet connection.
- **Machine Learning Model**: Utilizes MobileNetV3 for pet skin disease detection.
- **Custom Dataset Handling**: Balances class distribution for improved accuracy.
- **Efficient Computation**: Optimized for low-power devices.
- **Lightweight Deployment**: Can be integrated directly into mobile applications.

## Technology Stack
- **Language**: Python
- **Machine Learning**: TensorFlow
- **Model**: MobileNetV3 (Fine-tuned)
- **Data Processing**: NumPy, OpenCV
- **Logging & Evaluation**: Matplotlib, Seaborn, Scikit-learn

## Installation
### Prerequisites
- Python 3.x installed
- Virtual environment (recommended)
- Required dependencies

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/eru011/wellfur.git
   cd wellfur
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run training (Ensure dataset is placed inside `data/` folder):
   ```bash
   python train_wellfur.py
   ```

## Model Training
### Training Pipeline
The model follows these steps:
1. **Dataset Loading**: Images are loaded and split into training & validation sets.
2. **Preprocessing**: Images are resized and normalized for MobileNetV3.
3. **Model Fine-Tuning**: Pretrained MobileNetV3 is fine-tuned with frozen layers.
4. **Training Execution**: Model trains using categorical cross-entropy loss.
5. **Evaluation**: Metrics such as accuracy, F1-score, and confusion matrix are generated.
6. **Model Saving**: The trained model is saved as `train_wellfur.keras`.

## Future Improvements
- **Model Optimization**: Further optimize MobileNetV3 for better performance.
- **Expanded Dataset**: Improve accuracy by training on more diverse images.
- **Mobile Integration**: Seamlessly embed the model into the WellFur Flutter app.

## Contributors
- **Joachim Acosta** – Machine Learning Engineer

## Contact
For questions or contributions, feel free to reach out:
- Email: joachimacosta@gmail.com
- GitHub: [eru011](https://github.com/eru011)

---
Thank you for contributing to WellFur! 🐶🐱

