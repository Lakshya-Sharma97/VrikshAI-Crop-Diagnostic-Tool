VrikshAI: Intelligent Crop Diagnostic System

VrikshAI is an advanced computer vision solution designed to assist farmers in early-stage plant disease detection. Built using a specialized deep learning pipeline, the system provides real-time diagnostics and actionable agricultural remedies to minimize crop loss and improve yield.
🛠️ Data Engineering & Dataset

The core of VrikshAI is trained on a massive, diverse dataset to ensure high classification accuracy across various agricultural environments.

    Dataset Scope: The training pipeline utilizes a comprehensive dataset consisting of over 87,000 high-resolution images.

    Class Diversity: The model is trained to recognize 87 distinct classes, including specific crop diseases, healthy plant states, and an anomaly detection class (ZZZ_Unknown) for robust real-world performance.

    Storage & Accessibility: Due to size constraints (approx. 9GB), the full image dataset is hosted externally.

        Access the Dataset: https://www.kaggle.com/datasets/lakshyasharma97/vrikshai-crop-disease-and-anomaly-detection-dataset

💻 Hardware Optimization

To ensure the system is both performant and accessible, the inference engine was specifically optimized for consumer-grade hardware.

    Model Architecture: Utilizes MobileNetV2, chosen for its depthwise separable convolutions which provide a significant reduction in parameter count without compromising accuracy.

    Inference Hardware: The application is optimized to run on an NVIDIA GeForce GTX 1650 GPU (4GB VRAM).

    Performance: By leveraging the MobileNetV2 architecture on the GTX 1650, the system achieves low-latency inference, making it suitable for real-time deployment on laptop or edge-based workstations.

🚀 Key Features

    Real-time Diagnostics: Fast image processing via a Streamlit-based web interface.

    Actionable Advice: Integrated Remedies Database that provides specific chemical or biological solutions for detected diseases.

    Cross-Platform Logic: Developed in Python using PyTorch, ensuring the codebase is modular and easy to scale.
