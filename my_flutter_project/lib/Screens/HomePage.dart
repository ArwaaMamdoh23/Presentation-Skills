import 'package:flutter/material.dart';
import 'dart:ui';
import '../widgets/custom_app_bar.dart'; 
import '../widgets/background_wrapper.dart'; 
import '../widgets/CustomDrawer .dart'; 
import 'package:my_flutter_project/Screens/AuthPage.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class HomePage extends StatefulWidget {    
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool _isUserSignedIn = false;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _checkAuthState();
  }

  Future<void> _checkAuthState() async {
    try {
      final session = await Supabase.instance.client.auth.currentSession;
      if (mounted) {
        setState(() {
          _isUserSignedIn = session != null;
          _isLoading = false;
        });
      }
    } catch (e) {
      print('Error checking auth state: $e');
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(
      appBar: CustomAppBar(
        showSignIn: !_isUserSignedIn,  // You can still keep showSignIn for other logic
        isUserSignedIn: _isUserSignedIn,
        hideSignInButton: true,  // Pass this to hide the button
      ),
      drawer: _isUserSignedIn ? CustomDrawer(isSignedIn: _isUserSignedIn) : null,
      backgroundColor: Colors.transparent,
      extendBodyBehindAppBar: true,
      body: BackgroundWrapper( // Wrap the entire page content
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                "Welcome to PresentSense\nEnhance Your Presentation Skills with AI!",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontFamily: 'MyCustomFont', // Apply the custom font
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  shadows: [
                    Shadow(
                      blurRadius: 3.0,
                      color: Color.fromARGB(137, 255, 255, 255),
                      offset: Offset(0, 0),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 40),  // Add space between the text and the button

              // Add the "Get Started" button here
              ElevatedButton(
                onPressed: () {
                  // Navigate to AuthPage when the "Get Started" button is pressed
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => const AuthPage()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color.fromARGB(255, 71, 41, 6).withOpacity(0.5), // Set background color
                  padding: const EdgeInsets.symmetric(horizontal: 50, vertical: 15),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
                child: const Text(
                  "Get Started",
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}