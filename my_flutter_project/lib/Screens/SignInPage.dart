import 'package:flutter/material.dart';
import 'UploadVideo.dart'; // Import UploadVideoPage
import 'package:email_validator/email_validator.dart'; // Add email validation
import 'package:firebase_auth/firebase_auth.dart'; // Firebase Authentication
import 'package:cloud_firestore/cloud_firestore.dart'; // Firestore Database
import '../widgets/custom_app_bar.dart'; // Import Custom AppBar
import '../widgets/background_wrapper.dart'; // ✅ Import the wrapper
// import '../widgets/CustomDrawer .dart'; 

class SignInPage extends StatefulWidget {
  const SignInPage({super.key});

  @override
  _SignInPageState createState() => _SignInPageState();
}

class _SignInPageState extends State<SignInPage> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false; // ✅ Loading state

  Future<void> _signIn() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true); // Show loading indicator

    try {
      String email = _emailController.text.trim().toLowerCase();
      String password = _passwordController.text.trim();

      // Authenticate with Firebase
      UserCredential userCredential = await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      String userId = userCredential.user!.uid;

      // Check Firestore if user exists
      DocumentSnapshot userDoc = await FirebaseFirestore.instance.collection('User').doc(userId).get();

      if (userDoc.exists) {
        // User found, navigate to UploadVideoPage
        Navigator.pushReplacement(context, MaterialPageRoute(builder: (context) => const UploadVideoPage()));
      } else {
        // User not found in Firestore, sign out and show error
        await FirebaseAuth.instance.signOut();
        _showError('Account not found in database. Please sign up.');
      }
    } catch (e) {
      _showError(_handleAuthError(e));
    } finally {
      setState(() => _isLoading = false); // Hide loading indicator
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  String _handleAuthError(dynamic e) {
    if (e is FirebaseAuthException) {
      switch (e.code) {
        case 'user-not-found': return 'No account found with this email.';
        case 'wrong-password': return 'Incorrect password.';
        case 'invalid-email': return 'Invalid email format.';
        default: return 'Sign-In failed. Please try again.';
      }
    }
    return 'An unexpected error occurred. Please try again.';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true, // ✅ Allows background behind the AppBar
      appBar: CustomAppBar(
        showSignIn: false, // ✅ No Sign-In button on Sign-In page
        isUserSignedIn: false, // ✅ Ensure it's false since we're signing in
        hideSignInButton: true, // ✅ Explicitly hide Sign-In button
      ),
      // drawer: CustomDrawer(isSignedIn: false), // ✅ Sidebar (won't be visible until signed in)
      backgroundColor: Colors.transparent,
      body: BackgroundWrapper( // ✅ Apply background
        child: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Form(
              key: _formKey,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  const Text(
                    'Welcome',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 26,
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

                  const SizedBox(height: 40),
                  // Email Field with Validation
                  TextFormField(
                    controller: _emailController,
                    decoration: _inputDecoration('Email'),
                    style: const TextStyle(color: Colors.white),
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter your email';
                      } else if (!EmailValidator.validate(value)) {
                        return 'This is not the correct email format';
                      }
                      return null;
                    },
                  ),
                  const SizedBox(height: 20),

                  // Password Field with Validation
                  TextFormField(
                    controller: _passwordController,
                    decoration: _inputDecoration('Password'),
                    obscureText: true,
                    style: const TextStyle(color: Colors.white),
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter your password';
                      } else if (value.length < 8) {
                        return 'Password must be at least 8 characters long';
                      }
                      return null;
                    },
                  ),
                  const SizedBox(height: 20),

                  // Sign In Button
                  ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 280),
                    child: Container(
                      decoration: _buttonDecoration(),
                      child: ElevatedButton(
                        onPressed: _isLoading ? null : _signIn,
                        style: _buttonStyle(),
                        child: _isLoading
                            ? const CircularProgressIndicator(color: Colors.white)
                            : const Text(
                                'Sign In',
                                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
                              ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),

                  // Sign Up Navigation
                  TextButton(
                    onPressed: () => Navigator.pushNamed(context, '/sign-up'),
                    child: const Text(
                      'Don\'t have an account? Sign Up',
                      style: TextStyle(color: Colors.white),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  // Common Input Decoration
  InputDecoration _inputDecoration(String hint) {
    return InputDecoration(
      hintText: hint,
      hintStyle: const TextStyle(color: Colors.white70),
      filled: true,
      fillColor: Colors.white.withOpacity(0.2),
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(30), borderSide: BorderSide.none),
      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
    );
  }

  // Common Button Decoration
  BoxDecoration _buttonDecoration() {
    return BoxDecoration(
      gradient: LinearGradient(
        colors: [
          Colors.blueGrey.shade900,
          Colors.blueGrey.shade700,
        ],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ),
      borderRadius: BorderRadius.circular(30),
      boxShadow: [
        BoxShadow(
          color: Colors.lightBlue.withOpacity(0.4),
          blurRadius: 15,
          offset: const Offset(0, 5),
        ),
      ],
    );
  }

  // Common Button Style
  ButtonStyle _buttonStyle() {
    return ElevatedButton.styleFrom(
      minimumSize: const Size(280, 60),
      backgroundColor: Colors.transparent,
      shadowColor: Colors.transparent,
      padding: const EdgeInsets.symmetric(vertical: 20),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
    );
  }
}
