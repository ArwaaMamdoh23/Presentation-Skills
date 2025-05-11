import requests
import json
from datetime import datetime

def print_section(title, data, indent=0):
    """Helper function to print a section of feedback with proper formatting."""
    indent_str = "  " * indent
    print(f"\n{indent_str}{'=' * (len(title) + 4)}")
    print(f"{indent_str}  {title}")
    print(f"{indent_str}{'=' * (len(title) + 4)}")
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"\n{indent_str}  {key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    print(f"{indent_str}    {sub_key.replace('_', ' ').title()}: {sub_value}")
            elif isinstance(value, list):
                print(f"\n{indent_str}  {key.replace('_', ' ').title()}:")
                for item in value:
                    print(f"{indent_str}    • {item}")
            else:
                print(f"{indent_str}  {key.replace('_', ' ').title()}: {value}")
    else:
        print(f"{indent_str}  {data}")

def save_feedback_to_file(feedback_data):
    """Save the feedback to a JSON file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feedback_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, indent=2, ensure_ascii=False)
    
    return filename

def analyze_presentation(video_path, user_id):
    """Analyze a presentation video and provide comprehensive feedback."""
    try:
        # Open the video file to send with the request
        with open(video_path, 'rb') as f:
            files = {'video': f}
            data = {'user_id': str(user_id)}  # Add user_id to the request
            
            # Make the POST request to upload the video
            response = requests.post('http://127.0.0.1:5000/upload_video', files=files, data=data)
            
            if response.status_code == 200:
                feedback_data = response.json()
                
                # Print comprehensive feedback
                print("\n📊 PRESENTATION ANALYSIS REPORT")
                print("=" * 40)
                
                # Emotion Analysis
                print_section("Emotion Analysis", feedback_data['emotion_analysis'])
                
                # Gesture Analysis
                print_section("Gesture Analysis", feedback_data['gesture_analysis'])
                
                # Posture Analysis
                print_section("Posture Analysis", feedback_data['posture_analysis'])
                
                # Speech Analysis
                print_section("Speech Analysis", feedback_data['speech_analysis'])
                
                # Audience Interaction Analysis
                print_section("Audience Interaction Analysis", feedback_data['audience_interaction'])
                
                # Save feedback to file
                saved_file = save_feedback_to_file(feedback_data)
                print(f"\n✅ Feedback saved to: {saved_file}")
                
                return feedback_data
            else:
                print("❌ Failed to upload video. Status code:", response.status_code)
                print("Error details:", response.text)
                return None
                
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return None

if _name_ == "_main_":
    # You can change the video path here
    video_path = 'Videos/TedTalk.mp4'
    user_id = 'test_user'  # Add a test user ID
    
    print(f"🎥 Analyzing presentation video: {video_path}")
    feedback = analyze_presentation(video_path, user_id)
    
    if feedback:
        print("\n✨ Analysis complete! Check the feedback above for detailed insights.")