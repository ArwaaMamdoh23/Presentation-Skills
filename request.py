import requests

# Open the video file to send with the request
with open('Videos/TedTalk.mp4', 'rb') as f:
    files = {'video': f}

    # Make the POST request to upload the video
    response = requests.post('http://127.0.0.1:5000/upload_video', files=files)

    # Print the response from the server
    if response.status_code == 200:
        print("Success:", response.json())  # This should print the feedback response
    else:
        print("Failed to upload video. Status code:", response.status_code)
        print(response.text)
