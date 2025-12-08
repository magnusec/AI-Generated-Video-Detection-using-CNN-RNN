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
