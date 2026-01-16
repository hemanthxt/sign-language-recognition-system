import os
import pickle
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe is not installed!")
    print("Install it with: pip install mediapipe")
    sys.exit(1)


def create_dataset_from_images(dataset_path='./dataset/images for phrases'):
    """
    Process images in dataset folder using MediaPipe to extract hand landmarks
    and create training data.
    """
    if not os.path.exists(dataset_path):
        raise RuntimeError(f'Dataset path not found: {dataset_path}')
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1)
    
    data = []  # List to store hand landmarks data
    labels = []  # List to store corresponding labels
    label_names = []  # List to store label names
    
    # Get all class directories
    class_dirs = sorted([d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"\nFound {len(class_dirs)} sign categories:")
    for idx, class_name in enumerate(class_dirs):
        print(f"  {idx}: {class_name}")
    
    # Create label mapping
    label_map = {class_name: idx for idx, class_name in enumerate(class_dirs)}
    
    print("\nProcessing images...")
    processed_count = 0
    skipped_count = 0
    
    # Process each class directory
    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        class_label = label_map[class_name]
        
        # Get all image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\nProcessing '{class_name}' ({len(image_files)} images)...", end='')
        class_processed = 0
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    skipped_count += 1
                    continue
                
                # Convert to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = hands.process(img_rgb)
                
                if results and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract landmarks
                        data_aux = []
                        x_ = []
                        y_ = []
                        
                        for lm in hand_landmarks.landmark:
                            x_.append(lm.x)
                            y_.append(lm.y)
                        
                        # Normalize coordinates
                        for lm in hand_landmarks.landmark:
                            data_aux.append(lm.x - min(x_))
                            data_aux.append(lm.y - min(y_))
                        
                        data.append(data_aux)
                        labels.append(class_label)
                        label_names.append(class_name)
                        class_processed += 1
                        processed_count += 1
                        break  # Only process first hand detected
                else:
                    skipped_count += 1
                    
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                skipped_count += 1
                continue
        
        print(f" ✓ {class_processed} processed")
    
    hands.close()
    
    print(f"\n{'='*60}")
    print(f"Dataset Processing Complete!")
    print(f"  Total images processed: {processed_count}")
    print(f"  Images skipped (no hand detected): {skipped_count}")
    print(f"  Number of classes: {len(class_dirs)}")
    print(f"{'='*60}\n")
    
    return np.array(data), np.array(labels), label_map


def train_model(data, labels, label_map):
    """
    Train RandomForest classifier on the dataset
    """
    print("Training model...")
    
    # Ensure uniform feature length
    max_length = max(len(item) for item in data)
    data_padded = np.array([
        np.pad(item, (0, max_length - len(item)), 'constant') 
        if len(item) < max_length else item[:max_length]
        for item in data
    ])
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        data_padded, labels, test_size=0.2, shuffle=True, 
        stratify=labels, random_state=42
    )
    
    print(f"Training set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    
    # Train RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    
    # Evaluate
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Model Training Complete!")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*60}\n")
    
    # Detailed classification report
    print("Classification Report:")
    print("="*60)
    
    # Get unique labels actually present in test set
    unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    reverse_label_map = {v: k for k, v in label_map.items()}
    target_names = [reverse_label_map[i] for i in unique_labels]
    
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))
    
    return model, accuracy


def save_model(model, label_map, model_path='model_dataset.p', labels_path='labels.pickle'):
    """
    Save the trained model and label mappings
    """
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model}, f)
    print(f"✓ Model saved to: {model_path}")
    
    # Save label mappings
    with open(labels_path, 'wb') as f:
        pickle.dump(label_map, f)
    print(f"✓ Labels saved to: {labels_path}")


def main():
    print("="*60)
    print("Sign Language Model Training Pipeline")
    print("="*60)
    
    try:
        # Step 1: Create dataset from images
        data, labels, label_map = create_dataset_from_images()
        
        if len(data) == 0:
            print("ERROR: No data was extracted from the dataset!")
            print("Make sure images contain visible hands.")
            sys.exit(1)
        
        # Step 2: Train model
        model, accuracy = train_model(data, labels, label_map)
        
        # Step 3: Save model and labels
        save_model(model, label_map)
        
        print("\n" + "="*60)
        print("SUCCESS! Model is ready to use.")
        print("="*60)
        print("\nNext steps:")
        print("1. Run inference with: python InferenceDataset.py")
        print("2. The model will recognize these signs:")
        for label_name in sorted(label_map.keys()):
            print(f"   - {label_name}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
