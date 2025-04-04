import 'package:flutter/material.dart';
import 'package:my_flutter_project/Screens/AuthPage.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:my_flutter_project/widgets/custom_app_bar.dart';
import 'package:my_flutter_project/widgets/CustomDrawer .dart';
import 'package:my_flutter_project/Screens/UploadVideo.dart';
import 'package:my_flutter_project/Screens/ProfilePage.dart';
import 'package:my_flutter_project/Screens/EditProfilePage.dart';
import 'package:my_flutter_project/Screens/SignInPage.dart';
import 'package:my_flutter_project/Screens/SignUpPage.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _supabase = Supabase.instance.client;
  bool _isUserSignedIn = false;

  @override
  void initState() {
    super.initState();
    _checkAuthState();
  }

  Future<void> _checkAuthState() async {
    final session = _supabase.auth.currentSession;
    setState(() {
      _isUserSignedIn = session != null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppBar(
        showSignIn: !_isUserSignedIn,
        isUserSignedIn: _isUserSignedIn,
      ),
      drawer: _isUserSignedIn ? CustomDrawer(isSignedIn: _isUserSignedIn) : null,
      backgroundColor: Colors.transparent,
      extendBodyBehindAppBar: true,
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.blue.shade900,
              Colors.blue.shade700,
              Colors.blue.shade500,
            ],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                'Welcome to Video Upload App',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 20),
              if (!_isUserSignedIn) ...[
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => SignInPage()),
                    );
                  },
                  child: const Text('Sign In'),
                ),
                const SizedBox(height: 10),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => SignUpPage()),
                    );
                  },
                  child: const Text('Sign Up'),
                ),
              ] else ...[
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => UploadVideoPage()),
                    );
                  },
                  child: const Text('Upload Video'),
                ),
                const SizedBox(height: 10),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => ProfilePage()),
                    );
                  },
                  child: const Text('View Profile'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
