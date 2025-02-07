import 'package:flutter/material.dart';
import 'AuthPage.dart'; 
import 'SignInPage.dart'; 
import 'SignUpPage.dart';
import 'UploadVideo.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Presentation Skills',
      initialRoute: '/splash', // Set initial route
      routes: {
       '/': (context) => AuthPage(),

        '/sign-up': (context) => const SignUpPage(),
        '/sign-in': (context) => const SignInPage(),
      },
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue.shade400,
          primary: Colors.blue.shade500,
          secondary: Colors.grey.shade300,
        ),
        scaffoldBackgroundColor: Colors.blue.shade50, // Light blue background
        textTheme: const TextTheme(
          headlineLarge: TextStyle(color: Colors.black, fontSize: 32, fontWeight: FontWeight.bold),
          bodyLarge: TextStyle(color: Colors.black),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blue.shade600,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(30),
            ),
            padding: const EdgeInsets.symmetric(vertical: 15),
          ),
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  "Welcome to AI Presentation Feedback",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.pushReplacementNamed(context, '/auth'); // Navigate using named route
                  },
                  child: Text("Get Started"),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
