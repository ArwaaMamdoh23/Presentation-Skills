import 'package:flutter/material.dart';
import 'dart:ui';
import '../widgets/custom_app_bar.dart'; // Import Custom AppBar
import '../widgets/background_wrapper.dart'; // ✅ Import the wrapper
import '../widgets/CustomDrawer .dart'; 
import 'package:supabase_flutter/supabase_flutter.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  bool _isUserSignedIn = false;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _checkAuthState();
  }

  Future<void> _checkAuthState() async {
    try {
      final session = await Supabase.instance.client.auth.currentSession;
      if (mounted) {
        setState(() {
          _isUserSignedIn = session != null;
          _isLoading = false;
        });
      }
    } catch (e) {
      print('Error checking auth state: $e');
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return Scaffold(
      appBar: CustomAppBar(
        showSignIn: !_isUserSignedIn,
        isUserSignedIn: _isUserSignedIn,
      ),
      drawer: _isUserSignedIn ? CustomDrawer(isSignedIn: _isUserSignedIn) : null,
      backgroundColor: Colors.transparent,
      extendBodyBehindAppBar: true,
      body: BackgroundWrapper( // ✅ Wrap the entire page content
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                "Welcome to PresentSense\nEnhance Your Presentation Skills with AI!",
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  shadows: [
                    Shadow(
                      blurRadius: 3.0,
                      color: Colors.white54,
                      offset: Offset(0, 0),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
