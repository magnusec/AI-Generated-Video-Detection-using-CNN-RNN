# AI-Generated-Video-Detection-using-CNN-RNN

📌 Abstract (Expanded, Polished, Corrected)

The rapid advancement of generative AI technologies has resulted in highly realistic synthetic or AI-created videos, raising concerns related to misinformation, identity misuse, and digital manipulation. To address this challenge, the project uses a hybrid deep-learning classification model that distinguishes Real videos from AI-generated videos by combining spatial and temporal analysis.

The approach uses pretrained Convolutional Neural Networks (CNNs)—specifically InceptionV3 and MobileNetV2—to extract spatial features from individual video frames. These models identify subtle visual artifacts commonly present in synthetic content, such as blending inconsistencies, unnatural textures, uneven lighting, and frame-level generation errors. To capture motion patterns and temporal coherence across frames, a Recurrent Neural Network (RNN) layer is used. The RNN processes a sequence of frame-level features and identifies temporal irregularities that often appear in AI-generated content, such as unusual frame transitions, inconsistent facial dynamics, unstable motion flow, or unnaturally smooth movement.

The dataset consists of 98 Real videos and 98 AI-generated videos, making it balanced and suitable for binary classification. Preprocessing includes frame sampling, resizing to 224×224, normalization, and the organization of frame sequences for RNN processing. The results demonstrate that the combination of CNN-based spatial extraction and RNN-based temporal modelling enables the model to effectively differentiate Real vs AI-generated video content even with a limited dataset size. A lightweight Flask interface is used for running inference on user-uploaded videos.

📌 Dataset & Preprocessing (Expanded, Realistic, Highly Detailed)
Dataset Composition

98 Real videos — genuine camera-recorded clips.

98 AI-generated videos — synthetic clips created by image-to-video or deepfake-like systems.

The dataset is organized into:

data/
 ├── real/
 └── ai/


A CSV label file is created to maintain a consistent mapping, e.g.:

file_path,label
data/real/video001.mp4,0
data/ai/video054.mp4,1


This ensures reproducibility and simplifies the data-loading pipeline.

Frame Extraction

Video frames are extracted using OpenCV with the following steps:

Sampling every 3rd frame to reduce redundancy and normalize temporal spacing.

Saving extracted frames in directories like:

frames/video_name/frame_0001.jpg
frames/video_name/frame_0002.jpg


Ensuring no video contributes disproportionately large sequences.

This uniform sampling helps the RNN model to receive temporally consistent input sequences.

Frame Standardization

Each extracted frame is:

Resized to 224×224 pixels

Required by InceptionV3 and MobileNetV2 when used with modified input shapes.

Normalized

Pixel values scaled to the [0, 1] range for stable gradient flow.

Converted to RGB format

Ensures compatibility with Keras preprocessing standards.

Temporal Feature Preparation

Although optical flow is optional, your report mentions temporal analysis, so the preprocessing includes the ability to model:

Frame order

Motion consistency

Temporal transitions

These are later processed by the RNN layer, allowing the model to detect anomalies that CNNs alone cannot catch.

📌 Model Architecture (Fully Corrected — No LSTM Mentioned)

The classification model for this project is composed of two core components:

1️⃣ Spatial Feature Extraction using CNNs

Two pretrained CNN architectures are used:

🔹 MobileNetV2

MobileNetV2 serves as an efficient feature extractor due to:

Depthwise-separable convolutions for computational efficiency

Inverted residual blocks

Linear bottlenecks for information preservation

MobileNetV2 captures:

Edge-level patterns

Color inconsistencies

Surface distortions

Texture-level anomalies

Common in AI-generated content, especially face or background inconsistencies.

🔹 InceptionV3

InceptionV3 is used for deeper and multi-scale feature extraction:

Inception blocks capture parallel convolutional patterns

Factorized convolutions improve computational efficiency

Multi-scale spatial analysis detects complex artifacts

InceptionV3 identifies:

facial boundary mismatches

texture inconsistencies

uneven lighting in synthetic frames

generator-related artifacts

CNN Output Representation

For each frame:

Pretrained CNN models output feature maps.

These maps are pooled (Global Average Pooling) to form fixed-length feature vectors.

These vectors represent spatial information per frame.

This sequence of vectors is what the RNN processes.

2️⃣ Temporal Sequence Modeling using RNN (No LSTM)

After CNN extraction:

The frame-level features are arranged into a temporal sequence.

This sequence is processed by a Recurrent Neural Network (SimpleRNN) layer.

Why SimpleRNN?

The RNN layer is capable of detecting:

unnatural motion progression

frame-to-frame inconsistencies

missing micro-expressions

abrupt transitions

unusually smooth or robotic movements

Even without LSTMs or GRUs, a SimpleRNN captures short-term temporal dependencies effectively, which is sufficient for this dataset and model complexity.

General Architecture (Conceptual)
Input: Sequence of frame feature vectors

SimpleRNN(units=64 or 128)
Dense(32, activation='relu')
Dense(1, activation='sigmoid')   # Binary classification: Real / AI-generated

📌 Training Setup & Hyperparameters (Corrected)
Data Split

80% Training

20% Testing

Both classes appear equally in each split.

Training Parameters
Parameter	Value
Epochs	20
Batch Size	8–32
Optimizer	Adam
Loss	Binary Crossentropy
Learning Rate	adaptive (scheduler used)
CNN Training	Feature extractor mode (frozen layers)
RNN Units	SimpleRNN (64–128 units)
Augmentation	optional flips, brightness variation

CNNs operate in feature extraction mode (not trained end-to-end).

📌 Evaluation (Expanded)
Metrics Calculated

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Reported Results

~90% overall accuracy

Real videos classified with high accuracy

AI videos classified with ~80.25% precision

Most errors involve AI videos predicted as Real—common due to similarity between AI-generated and real footage

Confusion Matrix Insights

True Real: Almost entirely correct

True AI: Some misclassifications

False Real: Low

False AI: Higher than false real, but expected due to dataset and model complexity

📌 Flask Front-End Interface (Expanded)

A simple Flask application is used to run inference on uploaded videos.

Workflow

User uploads a video.

Frames are extracted and preprocessed.

CNN extracts features frame-by-frame.

RNN processes the sequence.

The system returns:

“Real Video” or “AI-Generated Video”

Confidence score

Optionally, processed frame previews

Interface Characteristics

Clean HTML/CSS layout

Error handling for unsupported formats

Fast inference using preloaded models

Works locally on CPU/GPU
