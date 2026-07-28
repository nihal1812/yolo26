AI Multi-Camera Behaviour Recognition & Theft Detection System

A distributed real-time computer vision platform for multi-camera behaviour recognition, identity tracking, and intelligent theft detection on edge devices.


Overview

Retail theft continues to be one of the largest sources of financial loss for businesses. Traditional surveillance systems rely heavily on manual monitoring, making them inefficient, difficult to scale, and prone to human error.

This project introduces a real-time AI-powered behavioural recognition platform capable of analysing multiple camera streams simultaneously, tracking customer identities across cameras, recognising suspicious behaviours, and generating intelligent alerts in real time.

The system has been designed for deployment on NVIDIA Jetson edge devices while maintaining low latency and high throughput.

Key Features
Real-time multi-camera perception
YOLO-based human detection
DeepSORT multi-object tracking
Cross-camera person re-identification
Global identity stitching
Behaviour recognition using temporal deep learning
Theft detection policy engine
Real-time alert generation
Distributed microservice architecture
TensorRT accelerated inference
Edge deployment on NVIDIA Jetson Orin NX
Modular and scalable pipeline
System Architecture
                     RTSP Cameras
                           │
                           ▼
                  GStreamer Pipeline
                           │
                           ▼
                   Perception Service
                (YOLO + DeepSORT)
                           │
                           ▼
               Identity Stitching Service
             (Cross-Camera ReID Engine)
                           │
                           ▼
                  Behaviour Analysis
             (Temporal Feature Extraction)
                           │
                           ▼
                  Behaviour Classifier
                 (3D CNN + MLP Network)
                           │
                           ▼
                    Policy Decision Engine
                           │
                           ▼
                 Incident Generation Service
                           │
                           ▼
                Dashboard • Alerts • Logging
Pipeline

The platform follows a distributed processing pipeline where every service performs a dedicated task.

RTSP Streams

↓

Video Decoding

↓

Object Detection

↓

Object Tracking

↓

Person Re-Identification

↓

Identity Stitching

↓

Temporal Feature Extraction

↓

Behaviour Recognition

↓

Policy Decision Engine

↓

Incident Generation

↓

User Interface
Technology Stack
AI & Machine Learning
PyTorch
TensorRT
ONNX
YOLO
DeepSORT
ResNet50 ReID
3D CNN
Multi-Layer Perceptron
Computer Vision
OpenCV
GStreamer
RTSP Streaming
CUDA Acceleration
Backend
Python
ZeroMQ
Docker
Linux
PM2
Hardware
NVIDIA Jetson Orin NX 16GB
Multiple RTSP Cameras
Core Algorithms
1. Person Detection

People are detected using a TensorRT-optimised YOLO model running directly on the Jetson GPU.

Responsibilities:

Human localisation
Confidence estimation
Real-time inference
Non-Maximum Suppression
2. Multi Object Tracking

Detected people are assigned persistent local identities using DeepSORT.

Responsibilities:

Kalman Filter prediction
Hungarian assignment
Track management
Occlusion handling
3. Person Re-Identification

Every tracked individual is converted into an appearance embedding using a ResNet-based feature extractor.

This enables recognition of the same individual after:

camera transitions
temporary occlusions
tracking failures
4. Global Identity Stitching

Instead of maintaining identities independently on each camera, all local identities are mapped into a global identity space.

Example:

Camera 1

Person #14

↓

Global ID = 2

↓

Camera 3

Person #7

↓

Global ID = 2

This creates continuous identity tracking across the entire store.

5. Temporal Behaviour Analysis

Every tracked customer generates a temporal feature sequence.

Examples include:

movement history
speed
dwell time
interaction duration
object possession state
pose features

These sequences form the input to the behaviour recognition network.

6. Behaviour Recognition

The temporal sequence is processed using deep learning models capable of understanding actions over time.

Example behaviours:

Walking
Browsing
Picking Product
Holding Product
Returning Product
Suspicious Behaviour
Theft
7. Decision Policy Engine

Behaviour predictions alone are insufficient for reliable theft detection.

A dedicated policy engine combines:

Behaviour confidence
Identity history
Temporal consistency
Evidence accumulation
Confidence thresholds
Multi-stage voting

This significantly reduces false positives.

8. Incident Generation

Only validated incidents are forwarded to the user interface.

Each alert contains:

Person identity
Camera information
Behaviour timeline
Confidence score
Evidence frames
Performance Optimisations

The platform has been engineered for real-time deployment.

Optimisations include:

TensorRT inference
FP16 execution
CUDA acceleration
Distributed processing
ZeroMQ communication
Multi-process architecture
GPU memory optimisation
Asynchronous execution
Efficient batching
Project Structure
project/

├── perception/
├── reid/
├── identity_stitcher/
├── behaviour/
├── policy/
├── ui/
├── dashboard/
├── docker/
├── configs/
├── datasets/
├── models/
├── scripts/
└── docs/
Future Improvements
Multi-store deployment
Cloud analytics dashboard
Active learning pipeline
Federated learning
Transformer-based behaviour recognition
LLM-assisted incident summarisation
Customer flow analytics
Heatmap generation
Inventory-aware theft detection
Research Areas

This project combines concepts from multiple disciplines:

Computer Vision
Deep Learning
Multi-Object Tracking
Person Re-Identification
Behaviour Understanding
Distributed Systems
Edge AI
Real-Time Systems
GPU Computing
Embedded AI
Acknowledgements

This project was developed as a real-time edge AI platform for intelligent retail surveillance, integrating state-of-the-art computer vision, deep learning, and distributed system design principles to deliver scalable, low-latency behavioural analytics on embedded hardware.
