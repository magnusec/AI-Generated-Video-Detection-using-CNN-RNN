# AI-Generated-Video-Detection-using-CNN-RNN

This project implements a deep-learning based video classification model that detects whether a given video is **Real** or **AI-Generated**. The approach uses pretrained Convolutional Neural Network (CNN) models—**InceptionV3** and **MobileNetV2**—for spatial feature extraction from individual video frames, combined with a **Simple Recurrent Neural Network (RNN)** layer for modeling short-term temporal patterns across sequences of frames.

A balanced dataset of **98 real videos** and **98 AI-generated videos** is used for training and evaluation. Each video is converted into a fixed-length sequence of frames, preprocessed, passed through CNN feature extractors, and then classified through the RNN-based temporal model. The final output predicts whether the video is real or synthetically generated.

The model is integrated with a lightweight **Flask web interface**, allowing users to upload a video and receive an instant classification result along with confidence scores.


## Abstract

The project applies a hybrid deep-learning approach to classify videos as **Real** or **AI-Generated** using a combination of spatial and temporal analysis. Pretrained CNN models—**InceptionV3** and **MobileNetV2**—are used as feature extractors to capture frame-level visual characteristics, such as texture patterns, color inconsistencies, lighting variations, and generation artifacts commonly found in synthetic content. 

To incorporate temporal information, the extracted per-frame feature vectors are arranged into ordered sequences and processed using a **Simple RNN** layer, allowing the model to learn short-term temporal relationships and detect irregular motion patterns or unnatural transitions that may appear in AI-generated videos.

The dataset consists of **98 real videos** and **98 AI-generated videos**, sampled into frames at a fixed interval and standardized to a uniform size before training. The combined CNN-RNN model demonstrates strong performance in distinguishing real footage from artificially generated content, achieving reliable accuracy with balanced class representation. A Flask-based interface is included for practical inference and user video uploads.


## Dataset & Preprocessing

The project uses a balanced dataset consisting of **98 Real videos** and **98 AI-Generated videos**. All videos are organized into separate folders:

data/
 ├── real/
 └── ai/

### Frame Extraction
Each video is converted into a sequence of frames using OpenCV. To maintain temporal consistency while reducing redundancy:

- Every 3rd frame is sampled from each video.
- Frames are stored in:
  frames/<video_name>/frame_0001.jpg
  frames/<video_name>/frame_0002.jpg
  ...

### Frame Standardization
Before being passed to the CNN models, each frame undergoes:

- Resizing to **224 × 224**
- RGB conversion
- Normalization to the **[0, 1]** pixel range

This ensures compatibility with InceptionV3 and MobileNetV2.

### Label Organization
A CSV file is generated to map each video to its label:

file_path,label
data/real/video01.mp4,0
data/ai/video45.mp4,1

This helps maintain consistent data loading for training and testing.

### Sequence Preparation
After frame extraction:

- Each frame is passed through the CNN feature extractor.
- Resulting frame-level feature vectors are combined into a sequence.
- These sequences serve as input to the Simple RNN layer for temporal modeling.

This pipeline ensures that each video contributes a uniform, properly processed feature sequence for the classification model.


## Model Architecture

The model used in this project combines spatial feature extraction from pretrained CNNs with temporal sequence modeling using a Simple RNN layer. The architecture operates in two main stages:

### 1. Spatial Feature Extraction (CNN)

Two pretrained CNN backbones are used for extracting frame-level features:

- **InceptionV3**  
  Used with `include_top=False` and Global Average Pooling. It captures multi-scale visual patterns and detects subtle artifacts such as texture inconsistencies, unnatural lighting, and boundary distortions often present in AI-generated frames.

- **MobileNetV2**  
  Also used with `include_top=False` and Global Average Pooling. It provides efficient feature extraction using depthwise separable convolutions and inverted residual blocks, capturing fine details with lower computational cost.

Each input frame is passed through one of these models to generate a fixed-length feature vector. These vectors represent the spatial characteristics of the frame.

### 2. Temporal Modeling (Simple RNN)

The sequence of feature vectors extracted from all frames of a video is fed into a **Simple RNN** layer. The RNN processes the ordered frame features and models short-term temporal dependencies, allowing the system to identify motion irregularities, unnatural transitions, or temporal inconsistencies that may appear in AI-generated content.

A typical structure of the temporal classifier includes:

- Simple RNN layer (e.g., 64–128 units)
- Dense layer with ReLU activation for feature refinement
- Final Dense layer with Sigmoid activation for binary classification (Real vs AI)

### 3. Output

The final output is a probability value between 0 and 1.  
- Values closer to **0** indicate **Real** videos.  
- Values closer to **1** indicate **AI-Generated** videos.

This architecture enables the model to analyze both spatial frame details and temporal frame progression, providing a reliable Real vs AI classification workflow.


## Training Setup & Hyperparameters

The model is trained using a balanced dataset of 98 Real and 98 AI-Generated videos. Each video is converted into a sequence of extracted frames, processed through the CNN feature extractors, and then passed to the Simple RNN classifier. The following setup and hyperparameters are used during training:

### Train/Test Split
- **80% Training**
- **20% Testing**
- Stratified to ensure equal representation of Real and AI-Generated videos in both sets.

### Training Hyperparameters
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Batch Size:** 8–32 (depending on available GPU/CPU resources)  
- **Epochs:** 20  
- **Learning Rate:** Adaptive, with ReduceLROnPlateau or similar scheduling  
- **Regularization:** Dropout layers included in the RNN and Dense units  
- **Shuffling:** Enabled during training to prevent sequence order bias  

### CNN Usage
- InceptionV3 and MobileNetV2 are used in **feature extraction mode** (frozen layers).  
- Global Average Pooling creates compact feature vectors for each frame.

### Sequence Configuration
- A fixed number of frames per video is used for training to maintain uniform sequence length.
- Frame features are stacked into arrays of shape:  
  **(sequence_length, feature_dimension)**  
- These sequences form the input to the Simple RNN layer.

### Augmentation
(Optional) Frame-level augmentation techniques may be applied to increase robustness:
- Horizontal flips  
- Minor rotations  
- Brightness adjustments  

### Model Saving
The trained model is saved in standard Keras `.h5` format, allowing easy loading for inference or integration with the Flask interface.

This training configuration ensures stable learning behavior and allows the model to effectively capture both spatial and temporal patterns needed for Real vs AI video classification.


## Evaluation & Results

The performance of the CNN + RNN model is evaluated using multiple metrics to ensure a clear understanding of how well the system distinguishes Real videos from AI-Generated videos. The dataset is balanced, so standard binary classification metrics provide reliable insights.

### Evaluation Metrics
The following metrics are used during testing:

- **Accuracy**  
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

These metrics help assess both overall model performance and class-specific behavior.

### Overall Performance
The model achieves strong results on the testing set, demonstrating effective classification of Real vs AI-Generated videos. Key outcomes include:

- **~90% overall accuracy**
- High correct detection rate for **Real videos**
- Approximately **80.25% correct classification** for **AI-Generated videos**
- Low rate of Real videos being misclassified
- Slightly higher rate of AI videos being predicted as Real

### Confusion Matrix Analysis
The confusion matrix reveals the following:

- **True Real:** Majority correctly classified  
- **True AI:** Most correctly classified, but with some misclassifications  
- **False Real:** Very few cases where Real videos are predicted as AI  
- **False AI:** More common, where AI-Generated videos are predicted as Real

This reflects the natural difficulty in detecting high-quality synthetic video content, which often mimics real-world motion and lighting with high precision.

### Reliability of Predictions
- Real videos tend to produce clear prediction scores closer to 0.  
- AI-generated videos typically produce higher output scores, though borderline cases occur due to subtle synthetic artifacts.

### Summary
The results demonstrate that the combined CNN-based spatial analysis and RNN-based temporal modeling provide a strong, reliable approach for detecting AI-generated videos. Despite the small dataset size, the model captures critical cues that differentiate real footage from synthetic content and performs consistently across evaluation metrics.


## Flask Interface (Demo System)

A lightweight **Flask web interface** is included in the project to allow users to upload a video and receive an instant Real vs AI classification result. The interface provides a simple and practical way to test the model without using command-line tools.

### Overview
The Flask application handles the complete inference workflow:

1. **Video Upload**  
   The user uploads an `.mp4`, `.avi`, or similar video file through the browser.

2. **Frame Extraction**  
   The uploaded video is processed using OpenCV to extract frames at fixed intervals.

3. **CNN Feature Extraction**  
   Each extracted frame is passed through the pretrained CNN model (InceptionV3 or MobileNetV2) to obtain spatial feature vectors.

4. **Sequence Assembly**  
   The frame-level features are converted into a fixed-length sequence suitable for the RNN model.

5. **RNN Classification**  
   The sequence is passed into the Simple RNN classifier to produce the final prediction score.

6. **Result Display**  
   The interface displays:
   - **Real** or **AI-Generated** label  
   - **Confidence score**  
   - Status messages for processing steps

### Interface Features
- Supports local video uploads
- Clean and simple HTML/CSS layout
- Automatic preprocessing and inference
- Fast response time due to preloaded model
- Error handling for invalid or corrupted video files

### Execution
To run the interface:
- Start the Flask server
- Open the provided URL in a browser
- Upload a video and view the prediction

The Flask interface provides a user-friendly, end-to-end demonstration of how the CNN + RNN model processes videos and generates real-time classification output.


