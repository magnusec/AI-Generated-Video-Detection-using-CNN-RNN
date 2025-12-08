# AI-Generated-Video-Detection-using-CNN-RNN

This project implements a deep-learning based video classification model that detects whether a given video is **Real** or **AI-Generated**. The approach uses pretrained Convolutional Neural Network (CNN) models—**InceptionV3** and **MobileNetV2**—for spatial feature extraction from individual video frames, combined with a **Simple Recurrent Neural Network (RNN)** layer for modeling short-term temporal patterns across sequences of frames.

A balanced dataset of **98 real videos** and **98 AI-generated videos** is used for training and evaluation. Each video is converted into a fixed-length sequence of frames, preprocessed, passed through CNN feature extractors, and then classified through the RNN-based temporal model. The final output predicts whether the video is real or synthetically generated.

The model is integrated with a lightweight **Flask web interface**, allowing users to upload a video and receive an instant classification result along with confidence scores.


## Abstract

The project applies a hybrid deep-learning approach to classify videos as **Real** or **AI-Generated** using a combination of spatial and temporal analysis. Pretrained CNN models—**InceptionV3** and **MobileNetV2**—are used as feature extractors to capture frame-level visual characteristics, such as texture patterns, color inconsistencies, lighting variations, and generation artifacts commonly found in synthetic content. 

To incorporate temporal information, the extracted per-frame feature vectors are arranged into ordered sequences and processed using a **Simple RNN** layer, allowing the model to learn short-term temporal relationships and detect irregular motion patterns or unnatural transitions that may appear in AI-generated videos.

The dataset consists of **98 real videos** and **98 AI-generated videos**, sampled into frames at a fixed interval and standardized to a uniform size before training. The combined CNN-RNN model demonstrates strong performance in distinguishing real footage from artificially generated content, achieving reliable accuracy with balanced class representation. A Flask-based interface is included for practical inference and user video uploads.
