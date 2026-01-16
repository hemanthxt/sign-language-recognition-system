# 📖 Indian Sign Language Detector - Complete User Guide

## 🎯 Available Signs (43 Total)

Your trained model can recognize these Indian Sign Language phrases:

### Everyday Actions & Requests
1. **again** - Repeat action
2. **agree** - Express agreement
3. **answer** - Provide response
4. **break** - Take a break
5. **change** - Modify/switch
6. **chat** - Conversation
7. **email** - Electronic mail
8. **file** - Document/folder
9. **join** - Come together
10. **meet** - Meeting/encounter
11. **open** - Unlock/reveal
12. **pass** - Go past/succeed
13. **practice** - Training/rehearsal
14. **remember** - Recall memory
15. **shift** - Move/change
16. **stop** - Halt
17. **wait** - Pause
18. **write** - Compose text

### Greetings & Expressions
19. **good morning** - Morning greeting
20. **happy birthday** - Birthday wishes
21. **congratulations** - Celebrate success
22. **keepsmile** - Stay positive

### Questions & Communication
23. **how are you** - Health inquiry
24. **where** - Location question
25. **questions** - Inquiries

### Needs & Feelings
26. **i need help** - Request assistance
27. **hungry** - Need food
28. **thirsty** - Need drink
29. **sick** - Unwell/ill
30. **pressure** - Stress/burden

### Work & School Related
31. **attendance** - Presence record
32. **book** - Reading material
33. **opinion** - View/perspective
34. **problem** - Issue/challenge
35. **seat** - Sitting place
36. **team** - Group/collective

### Safety & Caution
37. **careful** - Be cautious
38. **mistake** - Error

### Other Common Words
39. **home** - Residence
40. **please** - Polite request
41. **sun** - Solar star
42. **this** - Demonstrative
43. **together** - Unified/joint
44. **understand** - Comprehend

---

## 🚀 How to Use the System

### Option 1: Live Webcam Detection
**Run the detector to recognize signs in real-time:**

```powershell
.\mp_env\Scripts\python.exe InferenceDataset.py
```

**What happens:**
- Your webcam will open
- Show your hand sign to the camera
- The system will:
  - Draw your hand landmarks (skeleton)
  - Display the predicted sign name
  - Show confidence percentage (how sure it is)
  - Draw a green box around your hand

**Controls:**
- Press **'q'** or **'f'** to quit the detector

**Tips for best results:**
- Good lighting helps a lot
- Keep your hand clearly visible
- Position your hand in the center of the frame
- Hold the sign steady for 1-2 seconds
- Only one hand should be visible at a time

---

### Option 2: View Sign Examples
**See how each sign looks in the dataset:**

```powershell
.\mp_env\Scripts\python.exe ShowSignExamples.py
```

**What happens:**
- You'll be asked what you want to do:
  1. View example images (visual guide)
  2. Print list of all signs
  3. Both

**Visual Guide Controls:**
- Press **SPACE** or **ENTER** - Next sign
- Press **'q'** - Quit
- Press **'r'** - Restart from beginning

This shows you the actual images from your dataset so you can learn how to perform each sign correctly!

---

## 📊 Model Information

- **Accuracy:** 99.07%
- **Training Images:** 1,606 images
- **Total Categories:** 43 sign phrases
- **Model File:** `model_dataset.p`
- **Labels File:** `labels.pickle`

---

## 🔄 Retraining the Model

If you want to add more images or retrain:

```powershell
.\mp_env\Scripts\python.exe TrainOnDataset.py
```

This will:
1. Process all images in `dataset/images for phrases/`
2. Extract hand landmarks using MediaPipe
3. Train a new RandomForest model
4. Save to `model_dataset.p` and `labels.pickle`

---

## 📁 Project Structure

```
Sign-Language-Detector-Project-main/
├── InferenceDataset.py          # Live webcam detection
├── TrainOnDataset.py             # Train model on dataset
├── ShowSignExamples.py           # Visual guide for signs
├── model_dataset.p               # Trained model (99.07% accuracy)
├── labels.pickle                 # Label mappings
├── mp_env/                       # Python environment (use this!)
└── dataset/
    └── images for phrases/       # Your 43 sign categories
        ├── again/                # 40 images
        ├── agree/                # 40 images
        ├── answer/               # 40 images
        └── ...                   # (41 more categories)
```

---

## 🎓 Learning the Signs

### Recommended Learning Path:

1. **Start with the Visual Guide:**
   ```powershell
   .\mp_env\Scripts\python.exe ShowSignExamples.py
   ```
   Browse through all signs to see how they look

2. **Practice in Front of Webcam:**
   ```powershell
   .\mp_env\Scripts\python.exe InferenceDataset.py
   ```
   Try to recreate the signs you learned

3. **Check Your Accuracy:**
   The detector shows confidence percentage - aim for 90%+ confidence!

### Common Phrases to Learn First:
- **hello** → Use "good morning"
- **yes** → Use "agree"
- **no** → (Try "stop" or "disagree" if you add it)
- **thank you** → Use "please" (polite gesture)
- **help me** → Use "i need help"

---

## 🐛 Troubleshooting

### Camera not opening?
- Check if another application is using the webcam
- Close Zoom, Teams, or other video apps
- Run the script again

### Low confidence predictions?
- Ensure good lighting
- Position your hand clearly in frame
- Hold the sign steady
- Only show one hand at a time
- Make sure your hand gesture matches the training images

### Model not recognizing a sign?
- View the example for that sign: `ShowSignExamples.py`
- Try to match the hand position exactly
- Some signs may need specific hand orientations

---

## 💡 Quick Command Reference

| Command | Purpose |
|---------|---------|
| `.\mp_env\Scripts\python.exe InferenceDataset.py` | Live detection |
| `.\mp_env\Scripts\python.exe ShowSignExamples.py` | View sign examples |
| `.\mp_env\Scripts\python.exe TrainOnDataset.py` | Retrain model |

---

## 📝 Notes

- The "hungry" category had no usable training images (0 processed), so it might not work well
- Model was trained with 99.07% accuracy on the test set
- Uses MediaPipe for hand landmark detection
- RandomForest classifier with 100 trees
- Processes only the first detected hand if multiple hands are visible

---

## 🎉 Success Tips

1. **Good Lighting:** Use bright, even lighting
2. **Plain Background:** Avoid busy backgrounds
3. **Steady Hands:** Hold signs for 1-2 seconds
4. **One Hand:** Show only one hand at a time
5. **Practice:** Use ShowSignExamples.py to learn proper hand positions

---

Enjoy using your Indian Sign Language Detector! 🙌
