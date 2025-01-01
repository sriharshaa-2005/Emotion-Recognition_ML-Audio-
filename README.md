## Audio Emotion Recognition Using CNN and Mel Spectrograms ##

This project focuses on recognizing emotions from audio files using a Convolutional Neural Network (CNN). The audio features were extracted using Mel Spectrograms, which convert audio signals into visual representations for effective processing by CNNs. The model was trained, evaluated, and saved for deployment.

 Dataset Details:

Source: CREMA Dataset
Number of Audio Files: 7,442
Audio Format: .wav
Emotions Covered: Happy, Sad, Angry, Fearful, Neutral
Sampling Rate: 16,000 Hz

Technologies and Tools Used:

 Programming Language: Python
 Libraries: TensorFlow, Librosa, NumPy, Pandas, Matplotlib, TQDM, Scikit-learn
 Feature Extraction: Mel Spectrograms

Model Architecture: Convolutional Neural Network (CNN)

Deployment: Saved Model in .keras format

Data Preprocessing and Feature Extraction:

Data Loading: Audio files were loaded and read using TensorFlow and Librosa.

Feature Extraction: Mel Spectrograms were extracted from each audio file.

Feature Scaling: StandardScaler was applied to normalize features.

Data Splitting: Dataset split into 80% training and 20% testing.

Model Architecture

Input Layer: Mel Spectrogram (128x64)

Conv2D Layer: 32 filters, (3x3) kernel, ReLU activation

MaxPooling Layer: (2x2) Pooling

Flatten Layer: Converts matrix to vector

Dense Layer: 128 Neurons, ReLU Activation

Output Layer: 1 Neuron, Sigmoid Activation

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Metrics: Accuracy

📈 Training Results

Epochs: 5

Batch Size: 64
Training Accuracy: 100%
Validation Accuracy: 100%
Test Accuracy: 100%

 Evaluation Metrics

Precision: 1.00
Recall: 1.00
F1-Score: 1.00

Confusion Matrix: Showed perfect classification results.

 Model Deployment

The trained CNN model was saved as Audio_recognition.keras and can be reloaded for future predictions.
