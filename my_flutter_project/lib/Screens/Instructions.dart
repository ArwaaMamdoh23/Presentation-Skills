import 'package:flutter/material.dart';
import '../widgets/custom_app_bar.dart';
import '../widgets/background_wrapper.dart';
import '../widgets/CustomDrawer .dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:easy_localization/easy_localization.dart'; // Add this import

class Instructions extends StatelessWidget {
  const Instructions({super.key});

  @override
  Widget build(BuildContext context) {
    bool isUserSignedIn = true; 
    double screenWidth = MediaQuery.of(context).size.width;
    double screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
      extendBodyBehindAppBar: true,  // Ensures the content extends behind the app bar
      appBar: CustomAppBar(
        showSignIn: false, 
        isUserSignedIn: isUserSignedIn,
      ),
      drawer: CustomDrawer(isSignedIn: isUserSignedIn),
      body: BackgroundWrapper(
        child: SingleChildScrollView(  // Allow scrolling if content overflows
          padding: const EdgeInsets.all(16.0),  // Padding for better spacing
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,  // Center everything horizontally
            children: [
              // Add padding at the top to avoid overlap
              SizedBox(height: 100),  // Adjust this value based on your app bar height

              // Page Title (Centered)
              Text(
                'Instructions for PresentSense'.tr(), // Use .tr() for translation
                style: TextStyle(
                  color: Colors.white,
                  fontSize: screenWidth * 0.08,  // Dynamically adjust font size based on screen width
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,  // Ensures the text is centered
              ),
              SizedBox(height: 20),

              // Introductory Text
              Text(
                'Welcome to PresentSense.\n This tool is designed to help you evaluate and improve your presentation skills.'.tr(), // Use .tr() for translation
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: screenWidth * 0.045,  // Dynamically adjust font size based on screen width
                  height: 1.5,
                ),
                textAlign: TextAlign.center,  // Center the text
              ),
              SizedBox(height: 20),

              // Step 1: Upload your presentation
              Row(
                mainAxisAlignment: MainAxisAlignment.center,  // Center the icons and text
                children: [
                  FaIcon(FontAwesomeIcons.upload, color: Colors.white, size: screenWidth * 0.08), // Upload Icon
                  SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'Step 1: Upload your presentation file (video). Our system will analyze your presentation.'.tr(), // Use .tr() for translation
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: screenWidth * 0.04,  // Dynamically adjust font size based on screen width
                        height: 1.5,
                      ),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 20),

              // Step 2: Evaluation Process
              Row(
                mainAxisAlignment: MainAxisAlignment.center,  // Center the icons and text
                children: [
                  FaIcon(FontAwesomeIcons.search, color: Colors.white, size: screenWidth * 0.08), // Evaluation Icon
                  SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'Step 2: The system will evaluate various aspects such as speech fluency, posture, gestures, and eye contact.'.tr(), // Use .tr() for translation
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: screenWidth * 0.04,  // Dynamically adjust font size based on screen width
                        height: 1.5,
                      ),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 20),

              // Step 3: Receive Feedback
              Row(
                mainAxisAlignment: MainAxisAlignment.center,  // Center the icons and text
                children: [
                  FaIcon(FontAwesomeIcons.commentAlt, color: Colors.white, size: screenWidth * 0.08), // Feedback Icon
                  SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'Step 3: Receive detailed feedback on your performance to help you improve your presentation skills.'.tr(), // Use .tr() for translation
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: screenWidth * 0.04,  // Dynamically adjust font size based on screen width
                        height: 1.5,
                      ),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 20),

              // Final Note
              Text(
                'Make sure to practice and upload your presentations regularly to see improvements over time. Happy presenting!'.tr(), // Use .tr() for translation
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: screenWidth * 0.04,  // Dynamically adjust font size based on screen width
                  height: 1.5,
                ),
                textAlign: TextAlign.center,  // Center the final note text
              ),
              SizedBox(height: 40),

              // Additional Notes Section
              Text(
                'Additional Notes:'.tr(), // Use .tr() for translation
                style: TextStyle(
                  color: Colors.white,
                  fontSize: screenWidth * 0.06,  // Dynamically adjust font size based on screen width
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 10),
              Text(
                '- This system is designed to analyze various aspects of your presentation.\n'
                '- Ensure the video is of high quality, and with clear visibility of the speaker to facilitate accurate analysis and detection.\n'
                '- The video duration should not exceed 5 minutes to ensure effective evaluation and feedback.\n'
                '- The analysis may take a few minutes to process.'.tr(), // Use .tr() for translation
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: screenWidth * 0.04,  // Dynamically adjust font size based on screen width
                  height: 1.5,
                ),
                textAlign: TextAlign.center,  // Center the additional notes
              ),
            ],
          ),
        ),
      ),
    );
  }
}
