# Sign Language Detector Project - Complete Documentation

**Date:** January 7, 2026  
**Project Type:** Indian Sign Language Recognition System  
**Language:** Python

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Problem Statement](#problem-statement)
4. [Why This Project](#why-this-project)
5. [Project Structure](#project-structure)
6. [How to Run](#how-to-run)
7. [Adding/Removing Signs](#addingremoving-signs)
8. [System Limitations](#system-limitations)
9. [Model Information](#model-information)
10. [Recognized Signs](#recognized-signs)

---

## Project Overview

This is an **Indian Sign Language Detector** system that recognizes hand gestures in real-time using computer vision and machine learning. The system uses a webcam to capture hand movements, processes them using MediaPipe for hand landmark detection, and classifies them using a Random Forest machine learning model.

### Key Features
- **Real-time Detection** - Recognizes 43+ Indian Sign Language phrases
- **Voice Output** - Speaks detected signs using text-to-speech
- **Hand Landmark Visualization** - Draws skeleton on detected hands
- **Confidence Display** - Shows prediction confidence percentage
- **Session Logging** - Saves all detected signs to text file
- **Sentence Builder** - Builds sentences from detected signs
- **Statistics Tracking** - Counts sign frequency

---

## Technologies Used

### Programming Language
- **Python 3.x**

### Core Libraries

1. **MediaPipe**
   - Google's ML solution for hand landmark detection
   - Detects 21 hand keypoints in real-time
   - Version: Latest stable

2. **OpenCV (cv2)**
   - Computer vision library
   - Used for webcam capture and image processing
   - Handles frame resizing and display

3. **NumPy**
   - Numerical computing library
   - Array operations and mathematical functions
   - Data normalization

4. **Scikit-learn**
   - Machine learning library
   - Random Forest Classifier implementation
   - Train/test split and accuracy metrics

5. **Pickle**
   - Python object serialization
   - Model and data persistence

### Additional Libraries

- **pyttsx3** - Text-to-speech engine
- **win32com.client** - Windows SAPI for speech synthesis
- **Threading** - Asynchronous voice output
- **Matplotlib** - Data visualization
- **datetime & collections** - Logging and statistics

---

## Problem Statement

### The Problem
**Communication barrier between deaf/hearing-impaired individuals and people who don't understand sign language.**

### Challenges Addressed

1. **Limited Accessibility**
   - Not everyone knows sign language
   - Difficult for deaf/hearing-impaired to communicate with general public

2. **Real-Time Translation**
   - Need for instant sign language recognition
   - Conversion to text and speech

3. **Indian Sign Language (ISL) Focus**
   - ISL is different from American Sign Language (ASL)
   - Limited tools available for ISL

4. **Cost-Effective Solution**
   - Professional interpreters are expensive
   - Not always available when needed

### The Solution
This project provides a **free, real-time AI-powered system** that:
- Uses a simple webcam (no expensive hardware)
- Recognizes 43 Indian Sign Language phrases
- Converts signs to text AND voice output
- Works instantly without internet connection
- Helps bridge the communication gap

### Impact
- **For deaf/hearing-impaired:** Express themselves to anyone
- **For others:** Understand sign language without learning it
- **For education:** Learn and practice ISL
- **For accessibility:** Make public spaces more inclusive

---

## Why This Project

### 1. Social Impact
- Helps **millions of deaf/hearing-impaired people** communicate
- Makes society more **inclusive and accessible**
- Addresses a real-world problem affecting daily life

### 2. Technical Challenge
- Combines multiple **cutting-edge technologies**
- Great portfolio project showcasing **AI/ML skills**
- Real-time processing requirements

### 3. Practical Application
- **Real-time** system with immediate results
- Uses **common hardware** (just a webcam)
- Can be deployed in schools, hospitals, government offices
- Low-cost solution with high impact

### 4. Growing Field
- Accessibility technology is increasingly important
- **Aligns with SDG Goals** (Reduced Inequalities)
- Strong demand for assistive technologies

### 5. Learning Opportunity
Hands-on experience with:
- Image processing
- Feature extraction
- Model training
- Real-time inference
- Bridge between theory and practical application

### 6. Scalability
- Can expand to more signs
- Adaptable to different sign languages
- Can add sentence formation, grammar rules

---

## Project Structure

### Main Folder
**Location:** `Sign-Language-Detector-Project-main\Sign-Language-Detector-Project-main\`

### Key Files & Directories

```
Sign-Language-Detector-Project-main/
├── InferenceDataset.py       # Main detection script (webcam)
├── TrainOnDataset.py          # Model training script
├── ShowSignExamples.py        # View dataset examples
├── run.py                     # Python launcher
├── run.bat                    # Batch file launcher
├── model_dataset.p            # Trained Random Forest model
├── labels.pickle              # Label mappings
├── README.md                  # Project documentation
├── USER_GUIDE.md              # User instructions
├── dataset/                   # Training data
│   └── images for phrases/    # Sign images (43 categories)
│       ├── again/             # 40 images per sign
│       ├── agree/
│       ├── answer/
│       └── ...
└── mp_env/                    # Python virtual environment
    ├── Scripts/               # Python executable
    └── Lib/                   # Installed packages
```

---

## How to Run

### Prerequisites
- Windows OS (uses Windows SAPI for speech)
- Python 3.x with virtual environment
- Webcam
- Good lighting

### Method 1: Batch File (Easiest)
```batch
.\run.bat
```

### Method 2: Python Script
```bash
python run.py
```

### Method 3: Direct Execution
```bash
.\mp_env\Scripts\python.exe InferenceDataset.py
```

### Controls
- Press **'q'** or **'f'** to quit the detector
- Show hand signs clearly to the camera
- Hold signs steady for best results

### Tips for Best Results
- Use good lighting
- Keep hand clearly visible
- Position hand in center of frame
- Hold sign steady for 1-2 seconds
- Only one hand should be visible at a time

---

## Adding/Removing Signs

### Adding a New Sign

#### Step 1: Create Folder
Create a new folder with your sign name inside `dataset\images for phrases\`

```powershell
New-Item -Path ".\dataset\images for phrases\thank you" -ItemType Directory
```

#### Step 2: Add Images
- Add **30-40 images** of that sign to the folder
- Name them: `1.png`, `2.png`, `3.png`, etc.
- Take photos showing the hand sign clearly
- Use various angles and positions
- Good lighting is essential

#### Step 3: Retrain Model
```powershell
.\mp_env\Scripts\python.exe TrainOnDataset.py
```

This will:
- Process all images including your new sign
- Extract hand landmarks using MediaPipe
- Train a new Random Forest model
- Save the updated model to `model_dataset.p`

#### Step 4: Test
```powershell
.\run.bat
```

### Removing a Sign

#### Step 1: Delete Folder
```powershell
Remove-Item -Path ".\dataset\images for phrases\SIGN_NAME" -Recurse
```

#### Step 2: Retrain Model
```powershell
.\mp_env\Scripts\python.exe TrainOnDataset.py
```

The model will automatically update to exclude the removed sign.

### Important Notes
- **Image Quality:** Use clear, well-lit images with visible hands
- **Quantity:** More images (30-40) = better accuracy
- **Consistency:** Hold the sign steady during capture
- **Single Hand:** Only one hand should be visible per image
- **Background:** Clean backgrounds work best
- **Model Update:** Always retrain after adding/removing signs
- **Backup:** Keep a backup of your trained model before retraining

---

## System Limitations

### 1. Low Light Performance
**❌ NO - Struggles in low light**

- MediaPipe requires 50% detection confidence
- Needs clear visibility of hand landmarks
- Poor lighting prevents hand detection
- **Solution:** Use in well-lit rooms or add desk lamp

### 2. Two-Hand Gestures
**❌ NO - Only single hand supported**

- System configured for `max_num_hands=1`
- If two hands visible, it picks only one
- Cannot recognize two-handed signs properly
- **Major limitation:** Many ISL signs require two hands

### 3. Different Users
**✅ YES - Works for different users**

- Uses normalized hand landmark positions
- User-independent recognition
- Works across different hand sizes, skin tones
- However, accuracy may vary with very large/small hands

### Complete Limitations List

#### Technical Limitations
1. ⚠️ Single hand only - can't do two-handed signs
2. ⚠️ Static images only - no motion/gesture sequences
3. ⚠️ 43 signs limit - very small vocabulary
4. ⚠️ No sentence grammar - just word detection
5. ⚠️ Requires webcam - can't work without camera

#### Environmental Limitations
6. ⚠️ Good lighting required - fails in dim conditions
7. ⚠️ Clean background preferred - clutter may confuse
8. ⚠️ Hand must be visible - can't handle occlusions
9. ⚠️ Distance matters - hand too far/close won't work

#### Model Limitations
10. ⚠️ Random Forest - simpler model, less accurate than deep learning
11. ⚠️ Limited training data - only 40 images per sign
12. ⚠️ No confidence threshold filtering - may give false positives

### Recommended Improvements
- Add two-hand support (`max_num_hands=2`)
- Improve low-light handling (image enhancement)
- Use deep learning (CNN/LSTM) for better accuracy
- Add dynamic gesture recognition (motion sequences)
- Expand vocabulary to 200+ signs
- Add sentence formation logic

---

## Model Information

### Machine Learning Model
- **Algorithm:** Random Forest Classifier
- **Type:** Supervised Learning
- **Task:** Multi-class Classification

### Features
- **Input:** 42 normalized hand landmark coordinates
  - 21 hand keypoints × 2 dimensions (x, y)
  - Normalized relative to hand position
- **Output:** Sign label (1 of 43 classes)

### Training Process
1. **Data Collection:** Images from dataset folder
2. **Feature Extraction:** MediaPipe extracts hand landmarks
3. **Normalization:** Coordinates normalized to [0, 1]
4. **Train/Test Split:** Typically 80/20 split
5. **Model Training:** Random Forest with multiple decision trees
6. **Evaluation:** Accuracy score and classification report
7. **Persistence:** Model saved using pickle

### How It Works

#### Step 1: Capture
- Webcam captures video frames at 30 fps
- Frames resized to 1280×720 for processing

#### Step 2: Detection
- MediaPipe detects hand and extracts 21 landmarks
- Landmarks represent key points (fingertips, joints, etc.)

#### Step 3: Normalization
- Coordinates normalized relative to hand position
- Makes model hand-size and position independent

#### Step 4: Prediction
- Random Forest model classifies the gesture
- Returns predicted label and confidence score

#### Step 5: Output
- Displays label on screen with bounding box
- Speaks the sign using text-to-speech
- Logs detection to file with timestamp

---

## Recognized Signs

The system can recognize **43 Indian Sign Language phrases:**

### Everyday Actions & Requests (18 signs)
1. again - Repeat action
2. agree - Express agreement
3. answer - Provide response
4. break - Take a break
5. change - Modify/switch
6. chat - Conversation
7. email - Electronic mail
8. file - Document/folder
9. join - Come together
10. meet - Meeting/encounter
11. open - Unlock/reveal
12. pass - Go past/succeed
13. practice - Training/rehearsal
14. remember - Recall memory
15. shift - Move/change
16. stop - Halt
17. wait - Pause
18. write - Compose text

### Greetings & Expressions (4 signs)
19. good morning - Morning greeting
20. happy birthday - Birthday wishes
21. congratulations - Celebrate success
22. keepsmile - Stay positive

### Questions & Communication (3 signs)
23. how are you - Health inquiry
24. where - Location question
25. questions - Inquiries

### Needs & Feelings (6 signs)
26. i need help - Request assistance
27. hungry - Need food
28. thirsty - Need drink
29. sick - Unwell/ill
30. pressure - Stress/burden

### Work & School Related (6 signs)
31. attendance - Presence record
32. book - Reading material
33. opinion - View/perspective
34. problem - Issue/challenge
35. seat - Sitting place
36. team - Group/collective

### Safety & Caution (2 signs)
37. careful - Be cautious
38. mistake - Error

### Other Common Words (6 signs)
39. home - Residence
40. please - Polite request
41. sun - Solar star
42. this - Demonstrative
43. together - Unified/joint
44. understand - Comprehend

---

## System Requirements

### Hardware
- **Computer:** Windows PC/Laptop
- **Webcam:** Built-in or USB webcam
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** 500MB for project files

### Software
- **OS:** Windows 10/11
- **Python:** 3.7 or higher
- **Virtual Environment:** Included (mp_env/)

### Environment
- **Lighting:** Good, consistent lighting
- **Background:** Plain, uncluttered background preferred
- **Space:** Clear area for hand gestures

---

## Workflow Summary

### Training Workflow
```
Images → MediaPipe → Hand Landmarks → Normalization → Training → Model
```

1. Collect/prepare images in dataset folder
2. Run TrainOnDataset.py
3. MediaPipe extracts hand landmarks
4. Data normalized and split
5. Random Forest trained
6. Model saved to model_dataset.p

### Inference Workflow
```
Webcam → Detection → Landmarks → Model → Prediction → Display/Voice
```

1. Run InferenceDataset.py
2. Webcam captures frames
3. MediaPipe detects hands
4. Landmarks extracted and normalized
5. Model predicts sign
6. Result displayed and spoken

---

## Troubleshooting

### Camera Not Opening
- Close other apps using camera (Zoom, Teams)
- Check camera permissions in Windows Settings
- Try unplugging and reconnecting USB camera

### Low Accuracy
- Improve lighting conditions
- Hold sign steady for 1-2 seconds
- Ensure hand is clearly visible
- Retrain model with more/better images

### Model Not Found
- Run TrainOnDataset.py first
- Ensure model_dataset.p exists in project folder

### Voice Not Working
- Check Windows sound settings
- Ensure speakers/headphones connected
- Text-to-speech requires Windows SAPI

---

## Future Enhancements

### Planned Improvements
1. **Two-hand support** - Recognize signs requiring both hands
2. **Expanded vocabulary** - 200+ signs
3. **Deep learning** - CNN or LSTM for better accuracy
4. **Dynamic gestures** - Motion-based sign sequences
5. **Sentence grammar** - Build grammatically correct sentences
6. **Mobile app** - Android/iOS deployment
7. **Low-light mode** - Image enhancement for dim lighting
8. **Multi-language** - Support for other sign languages

### Research Directions
- Transfer learning with pre-trained models
- 3D hand pose estimation
- Continuous sign language recognition
- Real-time translator for conversations

---

## References & Resources

### Documentation
- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands
- OpenCV Documentation: https://docs.opencv.org/
- Scikit-learn: https://scikit-learn.org/

### Sign Language Resources
- Indian Sign Language Research
- Sign Language Datasets
- Accessibility Guidelines

---

## Contact & Support

For issues, questions, or contributions:
- Check USER_GUIDE.md for detailed usage instructions
- Review README.md for project overview
- Examine code comments for implementation details

---

**Document Created:** January 7, 2026  
**Project Version:** 1.0  
**Python Version:** 3.x  
**Status:** Active Development

---

*This project demonstrates the application of computer vision and machine learning to solve real-world accessibility challenges, making communication more inclusive for the deaf and hearing-impaired community.*
