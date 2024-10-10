# Real-Time Yoga Pose Classification using MediaPipe and TensorFlow

This project implements real-time yoga pose classification using a pre-trained neural network model and MediaPipe Pose (by Google) for human pose detection. The model is trained on a yoga pose dataset from Kaggle and is capable of detecting various yoga poses through a webcam feed.

## Features

- **MediaPipe Integration**: Utilizes Google's MediaPipe Pose model to extract 3D pose landmarks from the human body.
- **Real-Time Classification**: Predicts yoga poses in real-time using a trained deep learning model.
- **Efficient Execution**: Can run on low-end devices due to the computational efficiency of MediaPipe.
- **Dynamic Pose Tracking**: Provides accurate 3D pose estimation using a two-step process—detection and tracking.
- **Confidence Filtering**: Allows you to adjust a confidence threshold to discard low-certainty predictions.

## Project Overview

This project uses MediaPipe Pose to extract pose keypoints from a video stream (such as a webcam). These keypoints are normalized using a pre-trained scaler and passed to a deep learning model trained on yoga pose data. The predicted pose is displayed on the video along with the confidence level of the prediction.

The core components of the project are:

- MediaPipe: For pose detection and tracking.
- TensorFlow/Keras: For yoga pose classification using a neural network.
- Scikit-learn: For scaling and label encoding.

## About MediaPipe

MediaPipe Pose by Google is a powerful tool for pose detection that runs efficiently even on low-end devices. It uses a two-step process for pose estimation:

1. **Detection**: The initial detection identifies the human body in the frame.
2. **Tracking**: Once detected, tracking subsequent frames becomes computationally inexpensive.

The MediaPipe Pose model has three variations:

- BlazePose Heavy: Suitable for high-precision pose estimation.
- BlazePose Full: Balanced for speed and accuracy.
- BlazePose Lite: Optimized for low-end devices, providing faster execution with reduced accuracy.

MediaPipe helps in detecting 33 keypoints of the human body in 3D space, making it an ideal solution for complex pose tracking applications like yoga pose classification.

## Dataset

The model used in this project was trained on the [Yoga Pose Classification Dataset from Kaggle](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification). The dataset contains images of individuals performing various yoga poses, which are categorized into multiple classes. The dataset is preprocessed by extracting pose keypoints, scaling the features, and encoding the class labels for training.

## Installation

To run this project, you'll need to install the following dependencies:

```bash
pip install tensorflow
pip install mediapipe
pip install opencv-python
pip install scikit-learn
```

## Usage

1. Clone the Repository:

```bash
git clone https://github.com/NitinYadav1511/Realtime-Pose-Detection-and-Classification.git
cd Realtime-Pose-Detection-and-Classification
```

2. Download the Pre-trained Model and Scaler: Ensure the following files are present in the directory:

   - `best_model.keras` (Pre-trained Keras model)
   - `scaler.save` (Pre-trained scaler for keypoint normalization)
   - `label_encoder.save` (Pre-trained label encoder for class names)

3. Run the Pose Classification:

```bash
python main.py
```

This will launch your webcam and start detecting yoga poses in real-time.

Controls:
- Press `q` to quit the application.

## Project Files

- `main.py`: The main script that loads the model, scaler, label encoder, and runs the real-time pose classification.
- `best_model.keras`: The trained model file used for yoga pose classification.
- `scaler.save`: Scaler file for normalizing the input keypoints.
- `label_encoder.save`: Encodes the pose labels for the model output.
- `pose_keypoints.csv`: Contains pre-extracted keypoints data for reference.

## How It Works

1. **Pose Detection**: The MediaPipe Pose model detects keypoints from the human body, representing the 3D positions of various landmarks (e.g., shoulders, knees, etc.).

2. **Preprocessing**: These keypoints are normalized using a scaler before being fed into the neural network for prediction.

3. **Classification**: The trained neural network predicts the yoga pose class based on the input keypoints.

4. **Confidence Filtering**: If the model's confidence for a predicted pose is below a defined threshold, it will display "Unknown Pose."

5. **Pose Tracking**: MediaPipe uses a tracking-based approach to maintain high efficiency during real-time video feed processing.

## Model Training

The model was trained using TensorFlow on the Kaggle dataset mentioned earlier. Preprocessing involved extracting 3D pose keypoints using MediaPipe and scaling them before feeding them into the neural network. The model was trained for classification with a focus on achieving high accuracy for each pose class.

## Future Improvements

- Implementing more advanced post-processing techniques to improve pose classification.
- Extending the project to classify more yoga poses or other physical activities.

## Contributor

This repository is maintained by [Nitin Yadav](https://github.com/NitinYadav1511).

## License

This project is licensed under the MIT License.

## Credits

- MediaPipe by [Google](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [Yoga Pose Classification Dataset on Kaggle](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification)
- TensorFlow for building and training the classification model
