import pickle
import sys
import cv2
import numpy as np
import pyttsx3
from threading import Thread
import time
from datetime import datetime
from collections import Counter

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not available")
    sys.exit(1)


def run_inference(model_path='model_dataset.p', labels_path='labels.pickle'):
    """
    Run real-time sign language detection using webcam with voice output
    """
    # Skip pyttsx3 initialization - we'll use Windows SAPI directly
    engine = None
    print("✓ Text-to-speech initialized (using Windows SAPI)")
    
    # Variable to track last spoken sign and timing
    last_spoken_sign = None
    last_speak_time = 0
    speak_delay = 0.5  # Wait 0.5 seconds before speaking same sign again (very fast response)
    is_speaking = False  # Flag to prevent overlapping speech
    sign_stable_count = 0  # Count how many frames show same sign
    required_stable_frames = 1  # Speak immediately on first detection (instant response)
    
    # NEW FEATURES: Text logging, sentence builder, and statistics
    detected_signs = []  # List of all detected signs
    sentence = []  # Current sentence being built
    sign_counter = Counter()  # Count how many times each sign detected
    session_start = datetime.now()
    log_file = f"detection_log_{session_start.strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Create log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Sign Language Detection Session\n")
        f.write(f"Started: {session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
    
    print(f"✓ Logging to: {log_file}")
    
    def speak_sign(sign_text):
        """Speak the detected sign in a separate thread"""
        nonlocal is_speaking
        if not is_speaking:
            try:
                is_speaking = True
                print(f"🔊 Speaking: {sign_text}")
                # Use Windows SAPI for reliable speech
                try:
                    import win32com.client
                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    speaker.Speak(sign_text)
                    del speaker
                    print(f"✓ Voice output completed: {sign_text}")
                except Exception as e:
                    print(f"⚠️ Windows SAPI failed: {e}")
                    # Fallback to pyttsx3
                    if engine:
                        print(f"  Trying pyttsx3 fallback...")
                        engine.say(sign_text)
                        engine.runAndWait()
                        print(f"✓ pyttsx3 output completed: {sign_text}")
                    else:
                        print(f"❌ No voice engine available!")
            except Exception as e:
                print(f"❌ Speech error: {e}")
            finally:
                is_speaking = False
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        model = model_dict.get('model')
        print(f"✓ Model loaded from: {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found: {model_path}")
        print("Please run TrainOnDataset.py first to train the model.")
        sys.exit(1)
    
    # Load label mappings
    try:
        with open(labels_path, 'rb') as f:
            label_map = pickle.load(f)
        # Create reverse mapping (index -> name)
        labels_dict = {v: k for k, v in label_map.items()}
        print(f"✓ Labels loaded: {len(labels_dict)} signs")
    except FileNotFoundError:
        print(f"ERROR: Labels file not found: {labels_path}")
        sys.exit(1)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(static_image_mode=False, 
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5,
                           max_num_hands=1)
    
    # Initialize webcam
    print("\nTrying to open webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows
    
    # Give camera time to initialize
    time.sleep(1)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        print("\nTroubleshooting:")
        print("1. Make sure no other application is using the camera")
        print("2. Check if your camera is connected")
        print("3. Try closing Zoom, Teams, or other video apps")
        sys.exit(1)
    
    print("✓ Webcam opened successfully")
    
    print("\n" + "="*60)
    print("Sign Language Detector Started!")
    print("="*60)
    print("Controls:")
    print("  - Press 'q' or 'f' to quit")
    print("  - Show your hand signs to the camera")
    print("  - Voice announces ALL detected signs (>10% confidence)")
    print("  - VERY SENSITIVE - detects even slight gestures!")
    print("="*60 + "\n")
    
    try:
        while True:
            data_aux = []
            x_ = []
            y_ = []
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Resize for better performance
            frame = cv2.resize(frame, (1280, 720))
            H, W, _ = frame.shape
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = hands.process(frame_rgb)
            
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Extract features from first hand
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)
                    
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))
                    
                    break  # Only use first hand
                
                # Make prediction if we have data
                if len(data_aux) > 0:
                    try:
                        # Ensure correct feature length
                        n_features = model.n_features_in_
                        if len(data_aux) < n_features:
                            data_aux.extend([0] * (n_features - len(data_aux)))
                        elif len(data_aux) > n_features:
                            data_aux = data_aux[:n_features]
                        
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_label = labels_dict.get(int(prediction[0]), 'Unknown')
                        
                        # Get prediction confidence
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba([np.asarray(data_aux)])
                            confidence = np.max(proba) * 100
                        else:
                            confidence = 100
                        
                        # Track sign stability - count consistent detections
                        current_time = time.time()
                        
                        # Speak if:
                        # 1. NEW sign detected (immediate voice output)
                        # 2. Same sign but enough time passed (avoid rapid repetition)
                        is_new_sign = predicted_label != last_spoken_sign
                        time_passed = current_time - last_speak_time > speak_delay
                        
                        
                        # Debug logging
                        if is_new_sign:
                            print(f"🆕 New sign detected: {predicted_label} ({confidence:.1f}%)")
                            print(f"  📊 Confidence: {confidence:.1f}% | Speaking: {is_speaking}")
                        
                        should_speak = (
                            confidence > 10 and 
                            not is_speaking and 
                            (is_new_sign or time_passed)
                        )
                        
                        if should_speak:
                            print(f"  ✅ Voice trigger activated for: {predicted_label}")
                            # Speak in background thread to avoid blocking
                            Thread(target=speak_sign, args=(predicted_label,), daemon=True).start()
                            last_spoken_sign = predicted_label
                            last_speak_time = current_time
                            
                            # NEW: Log detected sign to file
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"[{timestamp}] {predicted_label} ({confidence:.1f}%)\n")
                            
                            # NEW: Add to sentence builder
                            sentence.append(predicted_label)
                            detected_signs.append((timestamp, predicted_label, confidence))
                            sign_counter[predicted_label] += 1
                            
                            # Keep sentence to last 5 signs
                            if len(sentence) > 5:
                                sentence.pop(0)
                        
                        # Display current prediction on screen (for debugging)
                        display_text = f"{predicted_label} ({confidence:.1f}%)"
                        if is_new_sign:
                            display_text += " [NEW]"
                        
                        # Calculate bounding box
                        x1 = int(min(x_) * W) - 20
                        y1 = int(min(y_) * H) - 20
                        x2 = int(max(x_) * W) + 20
                        y2 = int(max(y_) * H) + 20
                        
                        # Ensure box is within frame
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W, x2), min(H, y2)
                        
                        # Draw prediction box - green if speaking, yellow otherwise
                        box_color = (0, 255, 255) if is_speaking else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                        
                        # Add text background
                        text = f"{predicted_label} ({confidence:.1f}%)"
                        if is_speaking:
                            text = f"OUTPUT: {text}"  # Indicate when speaking
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        cv2.rectangle(frame, 
                                    (x1, y1 - text_size[1] - 10), 
                                    (x1 + text_size[0] + 10, y1), 
                                    box_color, -1)
                        
                        cv2.putText(frame, text, (x1 + 5, y1 - 8), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    except Exception as e:
                        print(f"Prediction error: {e}")
            
            # NEW: Display sentence builder at top of frame
            if sentence:
                sentence_text = " -> ".join(sentence)
                cv2.rectangle(frame, (10, 10), (W - 10, 60), (50, 50, 50), -1)
                cv2.putText(frame, f"Sentence: {sentence_text}", (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # NEW: Display statistics at bottom of frame
            stats_y = H - 80
            cv2.rectangle(frame, (10, stats_y), (W - 10, H - 10), (50, 50, 50), -1)
            total_signs = len(detected_signs)
            unique_signs = len(sign_counter)
            elapsed = (datetime.now() - session_start).total_seconds()
            stats_text = f"Total: {total_signs} signs | Unique: {unique_signs} | Time: {int(elapsed)}s"
            cv2.putText(frame, stats_text, (20, stats_y + 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Most detected sign
            if sign_counter:
                most_common = sign_counter.most_common(1)[0]
                cv2.putText(frame, f"Most Used: {most_common[0]} ({most_common[1]}x)", 
                          (20, stats_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Sign Language Detector - Press Q to Quit', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('f'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        # NEW: Write session summary to log file
        session_end = datetime.now()
        duration = (session_end - session_start).total_seconds()
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Session Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Ended: {session_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {int(duration)} seconds\n")
            f.write(f"Total Signs Detected: {len(detected_signs)}\n")
            f.write(f"Unique Signs: {len(sign_counter)}\n\n")
            
            if sign_counter:
                f.write("Sign Frequency:\n")
                for sign, count in sign_counter.most_common():
                    f.write(f"  {sign}: {count}x\n")
            
            if sentence:
                f.write(f"\nFinal Sentence: {' -> '.join(sentence)}\n")
        
        print("\nDetector stopped.")
        print(f"\n{'='*60}")
        print(f"Session Statistics:")
        print(f"{'='*60}")
        print(f"Total Signs: {len(detected_signs)}")
        print(f"Unique Signs: {len(sign_counter)}")
        print(f"Duration: {int(duration)}s")
        print(f"Log saved to: {log_file}")
        print(f"{'='*60}")


if __name__ == '__main__':
    run_inference()