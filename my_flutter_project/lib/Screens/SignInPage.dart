import 'package:flutter/material.dart';
import 'UploadVideo.dart'; 
import 'package:email_validator/email_validator.dart'; 
import 'package:supabase_flutter/supabase_flutter.dart'; 
import '../widgets/custom_app_bar.dart'; 
import '../widgets/background_wrapper.dart'; 
import 'HomePage.dart'; 
import 'package:shared_preferences/shared_preferences.dart';

class SignInPage extends StatefulWidget {
  const SignInPage({super.key});

  @override
  _SignInPageState createState() => _SignInPageState();
}

class _SignInPageState extends State<SignInPage> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;
  bool _isPasswordVisible = false;  // Track the visibility of the password
  List<String> _savedEmails = [];

  // Get the Supabase client
  final SupabaseClient _supabase = Supabase.instance.client;

  @override
  void initState() {
    super.initState();
    _checkExistingSession();
    _loadSavedEmails();
  }

  Future<void> _loadSavedEmails() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      _savedEmails = prefs.getStringList('emails') ?? [];
    });
  }

  Future<void> _saveEmail(String email) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    _savedEmails.add(email);
    prefs.setStringList('emails', _savedEmails);
  }

  Future<void> _checkExistingSession() async {
    try {
      final session = await _supabase.auth.currentSession;
      if (session != null) {
        if (mounted) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => UploadVideoPage()),
          );
        }
      }
    } catch (e) {
      print('Error checking session: $e');
    }
  }

  Future<void> _signIn() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);

    try {
      final email = _emailController.text.trim().toLowerCase();
      final password = _passwordController.text.trim();

      // Sign in with Supabase
      final AuthResponse response = await _supabase.auth.signInWithPassword(
        email: email,
        password: password,
      );

      if (response.session != null) {
        if (mounted) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => UploadVideoPage()),
          );
        }
        _saveEmail(email);  // Save email after successful login
      } else {
        throw Exception('Sign in failed - no session returned');
      }
    } on AuthException catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(e.message),
          backgroundColor: Colors.red,
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(e.toString()),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: CustomAppBar(
        showSignIn: false,
        isUserSignedIn: false,
        hideSignInButton: true,
      ),
      backgroundColor: Colors.transparent,
      body: BackgroundWrapper(
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

                  // Email Field
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

                  // Password Field with show/hide functionality
                  TextFormField(
                    controller: _passwordController,
                    decoration: InputDecoration(
                      labelText: 'Password',
                      labelStyle: const TextStyle(color: Colors.white),
                      filled: true,
                      fillColor: Colors.white.withOpacity(0.2),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(30),
                        borderSide: BorderSide.none,
                      ),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
                      suffixIcon: IconButton(
                        icon: Icon(
                          _isPasswordVisible ? Icons.visibility_off : Icons.visibility,
                          color: Colors.white,
                        ),
                        onPressed: _togglePasswordVisibility,
                      ),
                    ),
                    obscureText: !_isPasswordVisible,  // Toggle visibility
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
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
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

  // Toggle Password Visibility
  void _togglePasswordVisibility() {
    setState(() {
      _isPasswordVisible = !_isPasswordVisible;
    });
  }

  // Common Input Decoration
  InputDecoration _inputDecoration(String hint) {
    return InputDecoration(
      hintText: hint,
      hintStyle: const TextStyle(color: Colors.white70),
      filled: true,
      fillColor: Colors.white.withOpacity(0.2),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(30),
        borderSide: BorderSide.none,
      ),
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
      backgroundColor: const Color.fromARGB(255, 71, 41, 6).withOpacity(0.5),
      shadowColor: Colors.transparent,
      padding: const EdgeInsets.symmetric(vertical: 20),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(30),
      ),
    );
  }
}
