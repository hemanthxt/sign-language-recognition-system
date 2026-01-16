"""
Visual Guide - Show example images for each sign language phrase
This script displays sample images from your dataset so you can see how to perform each sign
"""

import os
import cv2
import sys


def show_sign_examples(dataset_path='./dataset/images for phrases'):
    """
    Display example images for each sign category
    """
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    # Get all sign categories
    categories = sorted([d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))])
    
    print("="*70)
    print("SIGN LANGUAGE DETECTOR - Visual Guide")
    print("="*70)
    print(f"\nFound {len(categories)} sign categories\n")
    print("Controls:")
    print("  - Press SPACEBAR or ENTER to see next sign")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to restart from beginning")
    print("="*70)
    
    current_idx = 0
    
    while True:
        if current_idx >= len(categories):
            print("\n✓ You've seen all signs! Press 'r' to restart or 'q' to quit.")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                current_idx = 0
                continue
            else:
                current_idx = 0
                continue
        
        category = categories[current_idx]
        category_path = os.path.join(dataset_path, category)
        
        # Get first image from this category
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print(f"No images found for '{category}'")
            current_idx += 1
            continue
        
        # Load and display the image
        img_path = os.path.join(category_path, image_files[0])
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not load image: {img_path}")
            current_idx += 1
            continue
        
        # Resize for better viewing
        height, width = img.shape[:2]
        max_dimension = 800
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Add text overlay with sign name and instructions
        display_img = img.copy()
        
        # Add black background for text
        cv2.rectangle(display_img, (10, 10), (display_img.shape[1]-10, 120), (0, 0, 0), -1)
        cv2.rectangle(display_img, (10, 10), (display_img.shape[1]-10, 120), (0, 255, 0), 2)
        
        # Add text
        sign_text = f"Sign {current_idx + 1}/{len(categories)}: {category.upper()}"
        cv2.putText(display_img, sign_text, (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        instruction = "Press SPACE for next sign, 'Q' to quit"
        cv2.putText(display_img, instruction, (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Sign Language Examples', display_img)
        
        # Console output
        print(f"\n[{current_idx + 1}/{len(categories)}] Showing: '{category}'")
        print(f"     Images available: {len(image_files)}")
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            current_idx = 0
        else:  # Any other key (including space/enter) goes to next
            current_idx += 1
    
    cv2.destroyAllWindows()
    print("\n✓ Visual guide closed.")


def print_all_signs(dataset_path='./dataset/images for phrases'):
    """
    Print a complete list of all available signs
    """
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path not found: {dataset_path}")
        return
    
    categories = sorted([d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))])
    
    print("\n" + "="*70)
    print("COMPLETE LIST OF AVAILABLE SIGNS")
    print("="*70)
    print(f"\nYour model can recognize these {len(categories)} signs:\n")
    
    # Print in columns for better readability
    for i, category in enumerate(categories, 1):
        print(f"  {i:2d}. {category}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    print("\nWhat would you like to do?")
    print("1. View example images for each sign (visual guide)")
    print("2. Print complete list of all signs")
    print("3. Both")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == '1':
        show_sign_examples()
    elif choice == '2':
        print_all_signs()
    elif choice == '3':
        print_all_signs()
        input("\nPress Enter to start visual guide...")
        show_sign_examples()
    else:
        print("Invalid choice. Running visual guide...")
        show_sign_examples()
