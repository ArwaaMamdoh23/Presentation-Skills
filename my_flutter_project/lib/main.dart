import 'package:flutter/material.dart';
import 'Screens/AuthPage.dart';
import 'Screens/SignUpPage.dart';
import 'Screens/SignInPage.dart';
import 'Screens/HomePage.dart'; // ✅ Import Home Page
import 'Screens/SettingsPage.dart'; // Import Settings Page
import 'Screens/ProfilePage.dart'; // Import Profile Page;
import 'Screens/ReportsPage.dart';
import 'Screens/UploadVideo.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';  // ✅ Import Firebase options

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // ✅ Proper Firebase Initialization with Web Support
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  runApp(MyApp());
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
        '/home': (context) => HomePage(), // ✅ Add Home Page route
        '/auth': (context) => AuthPage(), 
        '/sign-up': (context) => SignUpPage(),
        '/sign-in': (context) => SignInPage(),
        '/settings': (context) => SettingsPage(), 
        '/profile': (context) => ProfilePage(), 
        '/report': (context) => ReportsPage(), 
        '/upload': (context) => UploadVideoPage(),
      },
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue.shade400,
          primary: Colors.blue.shade500,
          secondary: Colors.grey.shade300,
        ),
        scaffoldBackgroundColor: Colors.white, // ✅ Changed from transparent to white
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
