import requests

# Open the video file to send with the request
files = {'video': open('Videos/TedTalk.mp4', 'rb')}

# Make the POST request to upload the video
response = requests.post('http://127.0.0.1:5000/upload_video', files=files)

# Print the response from the server
print(response.json())  # This should print the feedback response
