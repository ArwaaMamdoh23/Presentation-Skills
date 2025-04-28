
audio_file_path = extract_audio_from_video(video_file_path)

# Print the generated audio file path
print(f"Audio extracted and saved at: {audio_file_path}")


# Load Whisper model
model = whisper.load_model("base")

# Load audio using librosa
waveform, sr = librosa.load(audio_file_path, sr=16000)
waveform = waveform / max(abs(waveform))  # Normalize

# Transcribe using Whisper
result = model.transcribe(audio=waveform)
transcription = result["text"]
detected_lang = result["language"]
print("Transcription:", transcription)

def correct_grammar(sentence, t5_model, tokenizer, max_length=512):
    corrected_text = ""
    i = 0
    while i < len(sentence):
        # Get the next chunk of the sentence
        chunk = sentence[i:i + max_length]
        # Prepare input with prefix "grammar: " for T5 model
        input_text = "grammar: " + chunk
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)

        # Generate corrected text
        outputs = t5_model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
        corrected_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Append the corrected chunk
        corrected_text += corrected_chunk + " "

        # Move to the next chunk
        i += max_length

    return corrected_text.strip()



corrected_sentence = correct_grammar(transcription, t5_model, tokenizer)

# Function to normalize text by removing punctuation and converting to lowercase
def normalize_text(text):
    # Remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text

corrected_sentence = normalize_text(corrected_sentence)
transcription = normalize_text(transcription)

def grammatical_score(original, corrected):
    original_normalized = normalize_text(original)
    corrected_normalized = normalize_text(corrected)

    diff = difflib.ndiff(original_normalized.split(), corrected_normalized.split())
    changes = list(diff)

    corrected_words = [word for word in changes if word.startswith('+ ')]
    removed_words = [word for word in changes if word.startswith('- ')]

    num_changes = len(corrected_words) + len(removed_words)
    total_words = len(original_normalized.split())

    error_ratio = num_changes / total_words if total_words > 0 else 0

    if error_ratio == 0:
        return 10
    elif error_ratio < 0.05:
        return 9
    elif error_ratio < 0.1:
        return 8
    elif error_ratio < 0.2:
        return 6
    else:
        return 4
grammar_score = grammatical_score(transcription, corrected_sentence)

def get_grammar_feedback(score):
    if score == 10:
        return "Perfect grammar! Keep up the great work! 🎯"
    elif score >= 9:
        return "Very good grammar! A little improvement could make it perfect. 👍"
    elif score >= 8:
        return "Good grammar! There are a few minor mistakes. Keep improving. 😊"
    elif score >= 6:
        return "Grammar needs improvement. Review some rules and practice more. 📝"
    else:
        return "Poor grammar. It might help to practice more and focus on sentence structure. 🚀"
def calculate_speech_pace(audio_file):
    # Use pydub to get the duration of the audio
    audio_segment = AudioSegment.from_file(audio_file)
    duration = len(audio_segment) / 1000  # seconds

    # Load audio waveform using librosa (no ffmpeg)
    waveform, sr = librosa.load(audio_file, sr=16000)
    waveform = waveform / max(abs(waveform))  # Normalize

    # Transcribe with Whisper using waveform
    result = model.transcribe(audio=waveform)
    transcription = result["text"]

    # Calculate pace
    word_count = len(transcription.split())
    minutes = duration / 60
    pace = word_count / minutes if minutes > 0 else 0

    # Feedback
    if 130 <= pace <= 160:
        pace_feedback = "Your pace is perfect."
    elif 100 <= pace < 130:
        pace_feedback = "You need to speed up a little bit."
    elif pace < 100:
        pace_feedback = "You are going very slow."
    elif 160 < pace <= 190:
        pace_feedback = "You need to slow down a little bit."
    else:
        pace_feedback = "You are going very fast."

    return pace, transcription, pace_feedback

# Check if result is None before unpacking
if result is None:
    print("Error: Could not process the audio.")
else:
    pace, transcription, pace_feedback = result

    if transcription:
        print("Transcription:", transcription)
        print(f"Your grammatical score was: {grammatical_score(transcription, corrected_sentence)}/10")
        print("Speech Pace (WPM):", pace)
        print("Feedback:", pace_feedback)
    else:
        print("No transcription available.")

/////////////////////////////////////////////////////////////////////////////////////////////////////////
def correct_with_languagetool(text, lang_code):
        url = "https://api.languagetoolplus.com/v2/check"
        params = {
            "text": text,
            "language": lang_code,
            "enabledOnly": False
        }
        response = requests.post(url, data=params)
        matches = response.json().get("matches", [])
        corrected_text = text
        for match in reversed(matches):
            offset = match["offset"]
            length = match["length"]
            replacement = match["replacements"][0]["value"] if match["replacements"] else ""
            corrected_text = corrected_text[:offset] + replacement + corrected_text[offset+length:]
        return corrected_text

    # Use it on the Whisper transcription
corrected_lt_sentence = correct_with_languagetool(transcription, detected_lang)
print("Corrected (via LanguageTool):", corrected_lt_sentence)




# Add speech analysis feedback (grammar, pace, fluency, pronunciation)
combined_feedback_report.append("\n--- Speech Analysis ---")
combined_feedback_report.append(f" Detected Language: {detected_lang}")
combined_feedback_report.append(f"Corrected Sentence (T5 for English): {corrected_sentence}")
combined_feedback_report.append(f"Corrected Sentence (LanguageTool for '{detected_lang}'): {corrected_lt_sentence}")
combined_feedback_report.append(f"Grammar Score: {grammar_score}/10")
combined_feedback_report.append(f"Corrected Sentence: {corrected_sentence}")
combined_feedback_report.append(f"Grammar Feedback: {get_grammar_feedback(grammar_score)}")
combined_feedback_report.append(f"Speech Pace: {pace} WPM")
combined_feedback_report.append(f"Speech Pace Feedback: {pace_feedback}")
combined_feedback_report.append(f"Fluency Score: {filler_score}/100")
combined_feedback_report.append(f"Fluency Feedback: {filler_feedback}")
combined_feedback_report.append(f"Filler Word Breakdown: {filler_counts}")
combined_feedback_report.append(f"Pronunciation Score: {final_pronunciation_score}/100")
combined_feedback_report.append(f"Pronunciation Feedback: {pronunciation_feedback}")
combined_feedback_report.append("\n--- Audience Interaction Feedback ---")
for line in response_feedback:
    combined_feedback_report.append(line)


# Print the entire combined feedback report
print("\n--- Comprehensive Feedback Report ---")
for line in combined_feedback_report:
    print(line)


# Install translation model from English to the detected language
def install_translation_model(from_code, to_code):
    available_packages = package.get_available_packages()
    matching_package = next(
        (pkg for pkg in available_packages if pkg.from_code == from_code and pkg.to_code == to_code),
        None
    )
    if matching_package:
        download_path = matching_package.download()
        package.install_from_path(download_path)
        print(f"  translation from: {from_code} → {to_code}")
    else:
        print(f"No translation available for {from_code} → {to_code}")

# Only translate if the language isn't English
if detected_lang != "en":
    install_translation_model("en", detected_lang)

    installed_languages = translate.get_installed_languages()
    from_lang = next((lang for lang in installed_languages if lang.code == "en"), None)
    to_lang = next((lang for lang in installed_languages if lang.code == detected_lang), None)

    if from_lang and to_lang:
        translator = from_lang.get_translation(to_lang)
        translated_feedback_report = [translator.translate(line) for line in combined_feedback_report]

        print("\n --- Translated Feedback Report ---")
        for line in translated_feedback_report:
            print(line)
    else:
        print("Translation not available. Feedback shown in English.")
    # Use LanguageTool API for grammar correction
