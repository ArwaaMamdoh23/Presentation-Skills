import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:typed_data';
import 'dart:convert'; // For base64 encoding/decoding
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

import '../widgets/custom_app_bar.dart';
import '../widgets/background_wrapper.dart';

class EditProfilePage extends StatefulWidget {
  const EditProfilePage({super.key});

  @override
  _EditProfilePageState createState() => _EditProfilePageState();
}

class _EditProfilePageState extends State<EditProfilePage> {
  TextEditingController nameController = TextEditingController();
  TextEditingController currentPasswordController = TextEditingController();
  TextEditingController newPasswordController = TextEditingController();
  TextEditingController confirmPasswordController = TextEditingController(); // Added for confirm password

  Uint8List? _imageBytes;
  String? userId;

  // Password validation states
  bool hasUppercase = false;
  bool hasNumber = false;
  bool hasSpecialChar = false;
  bool hasMinLength = false;
  bool showPasswordRequirements = false;
  bool passwordsMatch = true; // To track if passwords match

  @override
  void initState() {
    super.initState();
    _loadUserProfile();
  }

  /// ✅ Load user profile from Firestore
  void _loadUserProfile() async {
    User? user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No user signed in")),
      );
      return;
    }

    userId = user.uid;

    try {
      DocumentSnapshot userDoc =
          await FirebaseFirestore.instance.collection('User').doc(userId).get();

      if (userDoc.exists) {
        setState(() {
          nameController.text = userDoc['Name'] ?? '';
          String? imageString = userDoc['ProfileImage'];
          if (imageString != null) {
            _imageBytes = base64Decode(imageString);
          }
        });
      }
    } catch (e) {
      print("❌ Firestore Error: $e");
    }
  }

  /// ✅ Pick and store profile image
  Future<void> _pickImage() async {
    FilePickerResult? result =
        await FilePicker.platform.pickFiles(type: FileType.image);
    if (result != null) {
      setState(() {
        _imageBytes = result.files.first.bytes!;
      });
    }
  }

  /// ✅ Validate password requirements
  void _validatePassword(String password) {
    setState(() {
      hasUppercase = RegExp(r'[A-Z]').hasMatch(password);
      hasNumber = RegExp(r'[0-9]').hasMatch(password);
      hasSpecialChar = RegExp(r'[!@#$%^&*(),.?":{}|<>]').hasMatch(password);
      hasMinLength = password.length >= 8;
      showPasswordRequirements = password.isNotEmpty;

      // Check if passwords match
      if (confirmPasswordController.text.isNotEmpty) {
        passwordsMatch = newPasswordController.text == confirmPasswordController.text;
      }
    });
  }

  /// ✅ Validate confirm password
  void _validateConfirmPassword(String confirmPassword) {
    setState(() {
      passwordsMatch = newPasswordController.text == confirmPassword;
    });
  }

  /// ✅ Save profile changes to Firestore & FirebaseAuth
  Future<void> _saveUserProfile() async {
    User? user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    String newName = nameController.text.trim();
    String newPassword = newPasswordController.text.trim();
    String confirmPassword = confirmPasswordController.text.trim();
    String currentPassword = currentPasswordController.text.trim();
    String? base64Image = _imageBytes != null ? base64Encode(_imageBytes!) : null;

    // Input validation
    if (newName.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Name cannot be empty")),
      );
      return;
    }
    if (newPassword.isNotEmpty && currentPassword.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Current password is required to change password")),
      );
      return;
    }

    // Check password requirements if new password is provided
    if (newPassword.isNotEmpty) {
      _validatePassword(newPassword);
      if (!(hasUppercase && hasNumber && hasSpecialChar && hasMinLength)) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Password doesn't meet all requirements")),
        );
        return;
      }

      // Check if passwords match
      if (newPassword != confirmPassword) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("New password and confirm password do not match")),
        );
        return;
      }
    }

    try {
      // ✅ Re-authenticate user before updating password
      if (newPassword.isNotEmpty) {
        bool isAuthenticated = await _reauthenticateUser(user, currentPassword);
        if (!isAuthenticated) return;
      }

      // ✅ 1. Update Firestore
      await FirebaseFirestore.instance.collection('User').doc(user.uid).update({
        'Name': newName,
        if (base64Image != null) 'ProfileImage': base64Image,
      });

      // ✅ 2. Update Firebase Auth
      if (newPassword.isNotEmpty) {
        await user.updatePassword(newPassword);
      }

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Profile updated successfully!")),
      );

      Navigator.pop(context);
    } catch (e) {
      print("❌ Error updating profile: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error updating profile: $e")),
      );
    }
  }

  /// ✅ Re-authenticate user with current password
  Future<bool> _reauthenticateUser(User user, String currentPassword) async {
    if (currentPassword.isEmpty) return false;

    try {
      AuthCredential credential = EmailAuthProvider.credential(
        email: user.email!,
        password: currentPassword,
      );

      await user.reauthenticateWithCredential(credential);
      return true;
    } catch (e) {
      print("⚠️ Re-authentication failed: $e");
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Incorrect current password. Please try again.")),
      );
      return false;
    }
  }

  @override
  void dispose() {
    nameController.dispose();
    currentPasswordController.dispose();
    newPasswordController.dispose();
    confirmPasswordController.dispose(); // Dispose the new controller
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: CustomAppBar(
        showSignIn: false,
        isUserSignedIn: true,
      ),
      body: BackgroundWrapper(
        child: Container(
          color: Colors.black.withOpacity(0.5), // Add overlay for better readability
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Center(
                  child: Stack(
                    children: [
                      CircleAvatar(
                        radius: 50,
                        backgroundColor: Colors.transparent,
                        backgroundImage: _imageBytes == null
                            ? null
                            : MemoryImage(_imageBytes!),
                        child: _imageBytes == null
                            ? Text(
                                nameController.text.isNotEmpty
                                    ? nameController.text[0]
                                    : 'U',
                                style: const TextStyle(
                                    fontSize: 40, color: Colors.white),
                              )
                            : null,
                      ),
                      Positioned(
                        bottom: 0,
                        right: 0,
                        child: CircleAvatar(
                          radius: 20,
                          backgroundColor: Colors.white,
                          child: IconButton(
                            icon: const Icon(Icons.edit,
                                size: 16, color: Colors.blueAccent),
                            onPressed: _pickImage,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 20),
                _buildTextField(nameController, 'Full Name'),
                _buildTextField(currentPasswordController, 'Current Password', obscureText: true),
                _buildTextField(
                  newPasswordController,
                  'New Password',
                  obscureText: true,
                  onChanged: _validatePassword,
                ),
                _buildTextField(
                  confirmPasswordController,
                  'Confirm New Password',
                  obscureText: true,
                  onChanged: _validateConfirmPassword,
                ),
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
                        _buildPasswordRequirement('Passwords must match', passwordsMatch),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 30),
                _buildSaveButton(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSaveButton() {
    return Center(
      child: ElevatedButton(
        onPressed: _saveUserProfile,
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.blueGrey.shade900,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 15, horizontal: 50),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(30),
          ),
        ),
        child: const Text('Save Changes', style: TextStyle(fontSize: 18)),
      ),
    );
  }

  Widget _buildTextField(
    TextEditingController controller,
    String labelText, {
    bool obscureText = false,
    void Function(String)? onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20.0),
      child: TextField(
        controller: controller,
        obscureText: obscureText,
        onChanged: onChanged,
        decoration: InputDecoration(
          labelText: labelText,
          filled: true,
          fillColor: Colors.white.withOpacity(0.2),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(30),
            borderSide: BorderSide.none,
          ),
          contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
        ),
        style: const TextStyle(color: Colors.white),
      ),
    );
  }

  Widget _buildPasswordRequirement(String text, bool isValid) {
    return Row(
      children: [
        Icon(
          isValid ? Icons.check_circle : Icons.cancel,
          size: 16,
          color: isValid ? Colors.green : Colors.red,
        ),
        const SizedBox(width: 8),
        Text(
          text,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
          ),
        ),
      ],
    );
  }
}