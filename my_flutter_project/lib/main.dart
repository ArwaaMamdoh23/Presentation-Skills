import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart'; // Ensure this file exists

import 'Screens/AuthPage.dart';
import 'Screens/SignUpPage.dart';
import 'Screens/SignInPage.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized(); 

  try {
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform, // ✅ Correct initialization
    ); 
    debugPrint("✅ Firebase Initialized Successfully");
  } catch (e) {
    debugPrint("❌ Firebase Initialization Error: $e");
  }

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PresentSense',
      debugShowCheckedModeBanner: false,
      home: const AuthPage(),
      routes: {
        '/sign-up': (context) => const SignUpPage(),
        '/sign-in': (context) => const SignInPage(),
      },
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue.shade400,
          primary: Colors.blue.shade500,
          secondary: Colors.grey.shade300,
        ),
        scaffoldBackgroundColor: Colors.transparent,
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
      builder: (context, child) {
        return Stack(
          children: [
            // 🟢 Ensure the asset path is correct
            Container(
              decoration: const BoxDecoration(
                image: DecorationImage(
                  image: AssetImage('assets/images/back.png'),
                  fit: BoxFit.cover,
                ),
              ),
            ),
            if (child != null) child,
          ],
        );
      },
    );
  }
}
