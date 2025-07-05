from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os

def display_image(img_path, title="Image"):
    """Display image using OpenCV and Matplotlib."""
    if not os.path.exists(img_path):
        print(f"[Error] File not found: {img_path}")
        return
    
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def analyze_face(img_path):
    """Analyze face attributes such as age, gender, emotion, and race."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    try:
        result = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=True)
        return result[0] if isinstance(result, list) else result
    except Exception as e:
        print(f"[Error] Analysis failed: {e}")
        return None

def verify_faces(img1_path, img2_path):
    """Compare two faces and verify if they belong to the same person."""
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        raise FileNotFoundError("One or both image paths are invalid.")
    
    try:
        result = DeepFace.verify(img1_path, img2_path, enforce_detection=True)
        return result
    except Exception as e:
        print(f"[Error] Verification failed: {e}")
        return None

def print_analysis(analysis):
    """Display face analysis results in a readable format."""
    if not analysis:
        print("[Error] No analysis result to display.")
        return

    print("🔍 Face Analysis Result:")
    print(f"  - Age     : {analysis.get('age')}")
    print(f"  - Gender  : {analysis.get('gender')}")
    print(f"  - Emotion : {analysis.get('dominant_emotion')}")
    print(f"  - Race    : {analysis.get('dominant_race')}")

if __name__ == "__main__":
    # Change these paths based on your local files
    face_img_path = 'D:/Tugas/AI/Face_Recognition/images/image1.jpg'
    img1_path = 'D:/Tugas/AI/Face_Recognition/images/images/image1.jpg'
    img2_path = 'D:/Tugas/AI/Face_Recognition/images/images/image2.jpg'

    # Display face
    display_image(face_img_path, title="Analyzed Face")

    # Face analysis
    analysis_result = analyze_face(face_img_path)
    print_analysis(analysis_result)

    # Face verification
    verification_result = verify_faces(img1_path, img2_path)
    if verification_result:
        print("\n✅ Face Verification Result:")
        print(f"  - Is same person: {verification_result['verified']}")
        print(f"  - Distance      : {verification_result['distance']:.4f}")
        print(f"  - Confidence    : {verification_result['confidence']:.2f}%")
