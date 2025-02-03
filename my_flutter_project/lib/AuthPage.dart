import 'package:flutter/material.dart';
import 'SignUpPage.dart';
import 'SignInPage.dart';

class AuthPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  const Color.fromARGB(255, 169, 171, 172),  // Adjusted gradient for better contrast
                  const Color.fromARGB(255, 195, 213, 226),
                ],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
          ),
          Center(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 30.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  Text(
                    'PresentSense', // App name
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 50),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.pushNamed(context, '/sign-up');
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color.fromARGB(255, 189, 191, 191),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                      padding: EdgeInsets.symmetric(vertical: 20),
                    ),
                    child: Text(
                      'Sign Up',
                      style: TextStyle(
                        fontSize: 18,
                        color: const Color.fromARGB(255, 175, 177, 177),
                      ),
                    ),
                  ),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.pushNamed(context, '/sign-in');
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color.fromARGB(255, 189, 191, 191),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                      padding: EdgeInsets.symmetric(vertical: 20),
                    ),
                    child: Text(
                      'Sign In',
                      style: TextStyle(
                        fontSize: 18,
                        color: const Color.fromARGB(255, 175, 177, 177),
                      ),
                    ),
                  ),
                  SizedBox(height: 40),
                  Text(
                    "Login with Social Media", // Added text above the social media buttons
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 10),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Google Sign-In Button with custom image in Circle
                      ClipOval(
                        child: Container(
                          color: Colors.white, // Background color for the circle
                          child: Image.asset(
                            'assets/images/google_icon.png', // Path to Google icon
                            width: 50, // Set the size of the icon
                            height: 50, // Set the size of the icon
                            fit: BoxFit.cover, // Make sure the icon fits within the circle
                          ),
                        ),
                      ),
                      SizedBox(width: 10),
                      // Facebook Sign-In Button with custom image in Circle
                      ClipOval(
                        child: Container(
                          color: Colors.white, // Background color for the circle
                          child: Image.asset(
                            'assets/images/facebook_icon.png', // Path to Facebook icon
                            width: 50, // Set the size of the icon
                            height: 50, // Set the size of the icon
                            fit: BoxFit.cover, // Make sure the icon fits within the circle
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
