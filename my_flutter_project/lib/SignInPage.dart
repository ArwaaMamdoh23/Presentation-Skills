
import 'package:flutter/material.dart';

class SignInPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [const Color.fromARGB(255, 169, 171, 172), const Color.fromARGB(255, 195, 213, 226)],
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
                    'Welcome Back',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 30,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 40),
                  TextField(
                    decoration: InputDecoration(
                      hintText: 'Email',
                      filled: true,
                      fillColor: Colors.white.withOpacity(0.2),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    style: TextStyle(color: Colors.white),
                  ),
                  SizedBox(height: 20),
                  TextField(
                    decoration: InputDecoration(
                      hintText: 'Password',
                      filled: true,
                      fillColor: Colors.white.withOpacity(0.2),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    obscureText: true,
                    style: TextStyle(color: Colors.white),
                  ),
                  SizedBox(height: 20),
             ElevatedButton(
  onPressed: () {
    // Add Sign-In Logic here
  },
  style: ElevatedButton.styleFrom(
    backgroundColor: const Color.fromARGB(255, 189, 190, 191), // Set the background color of the button here
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(30), // Optional: Set the border radius of the button
    ),
    padding: EdgeInsets.symmetric(vertical: 15), // Optional: Adjust padding
  ),
  child: Text(
    'Sign In',
    style: TextStyle(fontSize: 18, color: const Color.fromARGB(255, 175, 177, 177)), // Set the font color here
  ),
),
SizedBox(height: 20),
TextButton(
  onPressed: () {
    Navigator.pushNamed(context, '/sign-up');
  },
  child: Text(
    'Don\'t have an account? Sign Up',
    style: TextStyle(color: Colors.white),
  ),
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

