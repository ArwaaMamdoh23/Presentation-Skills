import 'package:flutter/material.dart';
import 'dart:ui';
import '../widgets/custom_app_bar.dart'; // Import Custom AppBar
import '../widgets/background_wrapper.dart'; // ✅ Import the wrapper
import '../widgets/CustomDrawer .dart'; 

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    bool isUserSignedIn = false; // Simulate user authentication

    return Scaffold(
      appBar: CustomAppBar(
        showSignIn: !isUserSignedIn,
        isUserSignedIn: isUserSignedIn
        ), 
      drawer: isUserSignedIn ? CustomDrawer(isSignedIn: isUserSignedIn) : null,
      backgroundColor: Colors.transparent,
      extendBodyBehindAppBar: true,
      body: BackgroundWrapper( // ✅ Wrap the entire page content
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                "Welcome to PresentSense\nEnhance Your Presentation Skills with AI!",
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  shadows: [
                    Shadow(
                      blurRadius: 3.0,
                      color: Colors.white54,
                      offset: Offset(0, 0),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
