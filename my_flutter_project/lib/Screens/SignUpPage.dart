import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:my_flutter_project/Screens/SignInPage.dart';
import '../widgets/custom_app_bar.dart'; // Import Custom AppBar
import '../widgets/background_wrapper.dart'; // ✅ Import the wrapper
// import '../widgets/CustomDrawer .dart'; 
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';


class SignUpPage extends StatefulWidget {
  const SignUpPage({super.key});

  @override
  _SignUpPageState createState() => _SignUpPageState();
}

class _SignUpPageState extends State<SignUpPage> {
  final _formKey = GlobalKey<FormState>();
  final TextEditingController _fullNameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController = TextEditingController();
  final FocusNode _passwordFocusNode = FocusNode();

  bool hasUppercase = false;
  bool hasNumber = false;
  bool hasSpecialChar = false;
  bool hasMinLength = false;
  bool showPasswordRequirements = false;

  @override
  void initState() {
    super.initState();
    _passwordFocusNode.addListener(() {
      setState(() {
        showPasswordRequirements = _passwordFocusNode.hasFocus;
      });
    });
  }

  void _validatePassword(String value) {
    setState(() {
      hasUppercase = value.contains(RegExp(r'[A-Z]'));
      hasNumber = value.contains(RegExp(r'[0-9]'));
      hasSpecialChar = value.contains(RegExp(r'[@\$!%*?&#]'));
      hasMinLength = value.length >= 8;
    });
  }

  @override
  void dispose() {
    _passwordFocusNode.dispose();
    super.dispose();
  }

  @override
  @override
Widget build(BuildContext context) {
  return Scaffold(
    extendBodyBehindAppBar: true, // ✅ Allows background behind the AppBar
    appBar: CustomAppBar(
      showSignIn: false, // ✅ No Sign-In button on Sign-Up page
      isUserSignedIn: false, // ✅ User is not signed in at this stage
      hideSignInButton: true, // ✅ Explicitly hide the Sign-In button
    ),
    // drawer: CustomDrawer(isSignedIn: false), // ✅ Sidebar (will be hidden until signed in)

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
                  'Create Your Account',
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
                _buildTextField(_fullNameController, 'Full Name'),
                const SizedBox(height: 20),
                _buildEmailField(),
                const SizedBox(height: 20),
                _buildPasswordField(),

                Visibility(
                  visible: showPasswordRequirements,
                  child: Container(
                    margin: const EdgeInsets.only(top: 10),
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.8),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          "Password Requirements",
                          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 5),
                        _buildPasswordRequirement('At least one uppercase letter', hasUppercase),
                        _buildPasswordRequirement('At least one number', hasNumber),
                        _buildPasswordRequirement('At least one special character', hasSpecialChar),
                        _buildPasswordRequirement('Minimum 8 characters', hasMinLength),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 20),
                _buildTextField(_confirmPasswordController, 'Confirm Password', obscureText: true, validator: (value) {
                  if (value != _passwordController.text) {
                    return 'Passwords do not match';
                  }
                  return null;
                }),
                const SizedBox(height: 20),
                _buildSignUpButton(),
              ],
            ),
          ),
        ),
      ),
    ),
  );
}


  Widget _buildTextField(TextEditingController controller, String hintText, {bool obscureText = false, String? Function(String?)? validator}) {
    return TextFormField(
      controller: controller,
      decoration: InputDecoration(
        hintText: hintText,
        hintStyle: const TextStyle(color: Colors.white70),
        filled: true,
        fillColor: Colors.white.withOpacity(0.2),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(30),
          borderSide: BorderSide.none,
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
      ),
      obscureText: obscureText,
      style: const TextStyle(color: Colors.white),
      validator: validator ?? (value) {
        if (value == null || value.isEmpty) {
          return 'Please enter $hintText';
        }
        return null;
      },
    );
  }

  Widget _buildEmailField() {
    return TextFormField(
      controller: _emailController,
      decoration: InputDecoration(
        hintText: 'Email',
        hintStyle: const TextStyle(color: Colors.white70),
        filled: true,
        fillColor: Colors.white.withOpacity(0.2),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(30),
          borderSide: BorderSide.none,
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
      ),
      style: const TextStyle(color: Colors.white),
      validator: (value) {
        if (value == null || value.isEmpty) {
          return 'Please enter your email';
        }
        bool hasNumber = RegExp(r'[0-9]').hasMatch(value);
        bool isValidFormat = RegExp(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$').hasMatch(value);
        if (!hasNumber) {
          return 'Email must contain at least one number.';
        }
        if (!isValidFormat) {
          return 'This is not the correct email format.';
        }
        return null;
      },
    );
  }

  Widget _buildPasswordField() {
    return TextFormField(
      controller: _passwordController,
      focusNode: _passwordFocusNode,
      decoration: InputDecoration(
        hintText: 'Password',
        hintStyle: const TextStyle(color: Colors.white70),
        filled: true,
        fillColor: Colors.white.withOpacity(0.2),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(30),
          borderSide: BorderSide.none,
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
      ),
      obscureText: true,
      style: const TextStyle(color: Colors.white),
      onChanged: _validatePassword,
    );
  }

  Widget _buildPasswordRequirement(String text, bool isMet) {
    return Row(
      children: [
        Icon(
          Icons.check_circle, 
          color: isMet ? Colors.green : Colors.grey, // Green when met, grey otherwise
        ),
        const SizedBox(width: 5),
        Text(text, style: TextStyle(color: isMet ? Colors.green : Colors.white70)),
      ],
    );
  }
  Widget _buildSignUpButton() {
  return Column(
    children: [
      ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 280),
        child: Container(
          decoration: BoxDecoration(
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
          ),
          child: ElevatedButton(
            onPressed: () async {
              if (_formKey.currentState?.validate() ?? false) {
                await _registerUser();
              }
            },
            style: ElevatedButton.styleFrom(
              minimumSize: const Size(280, 60),
              backgroundColor: Colors.transparent,
              shadowColor: Colors.transparent,
              padding: const EdgeInsets.symmetric(vertical: 20),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(30),
              ),
            ),
            child: const Text(
              'Sign Up',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
            ),
          ),
        ),
      ),
      const SizedBox(height: 20),

      // ✅ "Already have an account? Sign In" Button
      TextButton(
        onPressed: () {
          Navigator.pushNamed(context, '/sign-in'); // ✅ Navigate to Sign-In Page
        },
        child: const Text(
          'Already have an account? Sign In',
          style: TextStyle(color: Colors.white),
        ),
      ),
    ],
  );
}
Future<void> _registerUser() async {
  try {
    UserCredential userCredential = await FirebaseAuth.instance.createUserWithEmailAndPassword(
      email: _emailController.text.trim(),
      password: _passwordController.text.trim(),
    );

    String userId = userCredential.user!.uid;

    // Save user info in Firestore
    await FirebaseFirestore.instance.collection('User').doc(userId).set({
      'Email': _emailController.text.trim(),
      'Name': _fullNameController.text.trim(),
      'Role': 'user',  // Default role
      'User_id': userId,
    });

    // Navigate to Sign-In page after successful signup
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const SignInPage()),
    );
  } catch (e) {
    print("Error during registration: $e");
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Registration failed: $e')),
    );
  }
}

}