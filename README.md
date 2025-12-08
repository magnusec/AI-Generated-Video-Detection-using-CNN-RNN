# AI-Generated-Video-Detection-using-CNN-RNN

This project implements a deep-learning based video classification model that detects whether a given video is **Real** or **AI-Generated**. The approach uses pretrained Convolutional Neural Network (CNN) models—**InceptionV3** and **MobileNetV2**—for spatial feature extraction from individual video frames, combined with a **Simple Recurrent Neural Network (RNN)** layer for modeling short-term temporal patterns across sequences of frames.

A balanced dataset of **98 real videos** and **98 AI-generated videos** is used for training and evaluation. Each video is converted into a fixed-length sequence of frames, preprocessed, passed through CNN feature extractors, and then classified through the RNN-based temporal model. The final output predicts whether the video is real or synthetically generated.

The model is integrated with a lightweight **Flask web interface**, allowing users to upload a video and receive an instant classification result along with confidence scores.
