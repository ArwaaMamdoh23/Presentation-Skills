
import 'package:flutter/material.dart';

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
                colors: [const Color.fromARGB(255, 169, 171, 172), const Color.fromARGB(255, 195, 213, 226),],
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
                      fontSize: 30,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 40),
                 ElevatedButton(
  onPressed: () {
    Navigator.pushNamed(context, '/sign-up');
  },
  style: ElevatedButton.styleFrom(
    backgroundColor: const Color.fromARGB(255, 189, 191, 191),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(30), // Optional: Rounded corners for the button
    ),
    padding: EdgeInsets.symmetric(vertical: 15), // Optional: Adjust padding
  ),
  child: Text(
    'Sign Up',
    style: TextStyle(fontSize: 18, color: const Color.fromARGB(255, 175, 177, 177)), // Set the font color here
  ),
),
SizedBox(height: 20),
ElevatedButton(
  onPressed: () {
    Navigator.pushNamed(context, '/sign-in');
  },
  style: ElevatedButton.styleFrom(
    backgroundColor: const Color.fromARGB(255, 189, 191, 191), // Set background color here for Sign In button
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(30), // Optional: Rounded corners for the button
    ),
    padding: EdgeInsets.symmetric(vertical: 15), // Optional: Adjust padding
  ),
  child: Text(
    'Sign In',
    style: TextStyle(fontSize: 18, color: const Color.fromARGB(255, 175, 177, 177)), // Set the font color here
  ),
)

                  
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

