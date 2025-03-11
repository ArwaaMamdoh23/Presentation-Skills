import 'package:flutter/material.dart';
import 'Screens/AuthPage.dart';
import 'Screens/SignUpPage.dart';
import 'Screens/SignInPage.dart';
import 'Screens/HomePage.dart'; // ✅ Import Home Page
import 'Screens/SettingsPage.dart'; // Import Settings Page
import 'Screens/ProfilePage.dart'; // Import Profile Page;
import 'Screens/ReportsPage.dart';
import 'Screens/UploadVideo.dart';


void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PresentSense', // App name
      debugShowCheckedModeBanner: false, // Remove the debug banner
      initialRoute: '/home',
      routes: {
        '/home': (context) => const HomePage(), // ✅ Add Home Page route
        '/auth': (context) => const AuthPage(),  // ❌ Removed const
        '/sign-up': (context) => const SignUpPage(), // ❌ Removed const
        '/sign-in': (context) => const SignInPage(), // ❌ Removed const
        '/settings': (context) => const SettingsPage(), // ✅ Define settings route
        '/profile': (context) => const ProfilePage(), // ✅ Define profile route
        '/report': (context) =>  ReportsPage(), // ✅ Define profile route
        '/upload': (context) => const UploadVideoPage(), // ✅ Define profile route
      },
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue.shade400,
          primary: Colors.blue.shade500,
          secondary: Colors.grey.shade300,
        ),
        scaffoldBackgroundColor: Colors.transparent, // Make scaffold background transparent
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
        ),
      ),
    );
  }
}
