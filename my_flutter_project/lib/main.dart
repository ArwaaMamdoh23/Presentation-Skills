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
        '/splash': (context) => SplashScreen(),
        '/auth': (context) => AuthPage(),
        '/sign-in': (context) => SignInPage(),
        '/sign-up': (context) => SignUpPage(),
        '/upload-video': (context) => UploadVideoPage(),
      },
    );
  }
}

class SplashScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          Image.asset(
            'assets/images/e304c83d-ad29-4296-b695-7cefbcd4a3d0.png', 
            fit: BoxFit.cover,
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
