# 🍏 Food Freshness Detection (AI + Android)

This project demonstrates an end-to-end **Food Freshness Detection System** — from training a Convolutional Neural Network (CNN) in Python to deploying a **TensorFlow Lite model** inside an **Android (Java)** app.  
It classifies images of food as **Fresh** or **Rotten** using your device camera.

---

## 🧠 Features

✅ Train a CNN using TensorFlow/Keras  
✅ Convert trained model to TensorFlow Lite (`.tflite`)  
✅ Run real-time predictions in an Android app  
✅ Offline and on-device inference (no internet required)  
✅ Lightweight and optimized for mobile CPUs  

---

## 📂 Project Structure

food-freshness/
├── training/
│ ├── train_freshness.py # Model training script
│ ├── convert_tflite.py # Convert .h5 → .tflite
│ ├── model_freshness.h5 # Trained Keras model
│ ├── model_freshness_int8.tflite # Converted TFLite model
│ ├── class_indices.json # Label mapping
│ └── dataset/ # Contains 'fresh/' and 'rotten/' image folders
│
├── android_app/
│ ├── app/ # Android source (Java)
│ │ ├── src/main/java/com/example/freshnessapp/MainActivity.java
│ │ ├── res/layout/activity_main.xml
│ │ └── assets/ # Place .tflite + class_indices.json here
│ └── build.gradle
│
├── Food_Freshness_Project_Final.docx # Project documentation
├── .gitignore
└── README.md
