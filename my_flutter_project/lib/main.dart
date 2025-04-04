import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart'; // ✅ Import Supabase
import 'Screens/AuthPage.dart';
import 'Screens/SignUpPage.dart';
import 'Screens/SignInPage.dart';
import 'Screens/HomePage.dart'; // ✅ Import Home Page
import 'Screens/SettingsPage.dart'; // Import Settings Page
import 'Screens/ProfilePage.dart'; // Import Profile Page;
import 'Screens/ReportsPage.dart';
import 'Screens/UploadVideo.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  

  // ✅ Initialize Supabase
  await Supabase.initialize(
    url: 'https://ohllbliwedftnyqmthze.supabase.co', // Replace with your Supabase project URL
    anonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9obGxibGl3ZWRmdG55cW10aHplIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDM1NDIxMzAsImV4cCI6MjA1OTExODEzMH0.XW1XNf7v3-JX94-1xJNgPM70t2qvZoEClyAab85ie1o', // Replace with your Supabase anonymous key
  );

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
