import 'package:flutter/material.dart';
import 'AuthPage.dart';
import 'SignUpPage.dart';
import 'SignInPage.dart';

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
      initialRoute: '/',
      routes: {
        '/': (context) => const AuthPage(),  // ❌ Removed const
        '/sign-up': (context) => const SignUpPage(), // ❌ Removed const
        '/sign-in': (context) => const SignInPage(), // ❌ Removed const
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
      builder: (context, child) {
        return Stack(
          children: [
            Container(
              decoration: const BoxDecoration(
                image: DecorationImage(
                  image: AssetImage('assets/images/back.png'), // Ensure this path is correct
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
