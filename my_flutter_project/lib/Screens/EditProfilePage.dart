import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:file_picker/file_picker.dart'; // File picker package
import 'dart:typed_data'; // For handling file bytes
import '../widgets/custom_app_bar.dart'; // ✅ Import Custom AppBar
import '../widgets/background_wrapper.dart'; // ✅ Import Background Wrapper
import '../widgets/CustomDrawer .dart'; 

class EditProfilePage extends StatefulWidget {
  const EditProfilePage({super.key});

  @override
  _EditProfilePageState createState() => _EditProfilePageState();
}

class _EditProfilePageState extends State<EditProfilePage> {
  TextEditingController nameController = TextEditingController();
  TextEditingController emailController = TextEditingController();
  TextEditingController phoneController = TextEditingController();
  TextEditingController passwordController = TextEditingController();
  Uint8List? _imageBytes; // Store image as bytes

  @override
  void initState() {
    super.initState();
    _loadUserProfile();
  }

void _loadUserProfile() async {
  SharedPreferences prefs = await SharedPreferences.getInstance();
  setState(() {
    nameController.text = prefs.getString('name') ?? '';
    emailController.text = prefs.getString('email') ?? '';
    phoneController.text = prefs.getString('phone') ?? '';
    passwordController.text = prefs.getString('password') ?? '';
    String? imageBytesString = prefs.getString('profile_image_bytes');
    if (imageBytesString != null) {
      _imageBytes = Uint8List.fromList(imageBytesString.codeUnits);
    }
  });
}
  Future<void> _pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(type: FileType.image);
    if (result != null) {
      PlatformFile file = result.files.first;
      setState(() {
        _imageBytes = file.bytes;
      });
      SharedPreferences prefs = await SharedPreferences.getInstance();
      prefs.setString('profile_image_bytes', String.fromCharCodes(_imageBytes!));
    }
  }

  void _saveUserProfile() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setString('name', nameController.text);
    prefs.setString('email', emailController.text);
    prefs.setString('phone', phoneController.text);
    prefs.setString('password', passwordController.text);
    if (_imageBytes != null) {
      prefs.setString('profile_image_bytes', String.fromCharCodes(_imageBytes!));
    }
    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    bool isUserSignedIn = true; // ✅ Change based on user authentication status
    return Scaffold(
      extendBodyBehindAppBar: true, // ✅ Extends content behind the AppBar
      appBar: CustomAppBar(
        showSignIn: false, 
        isUserSignedIn: true, // ✅ Ensures Profile & Settings icons appear
      ),
      drawer: CustomDrawer(isSignedIn: isUserSignedIn), // ✅ Sidebar on the RIGHT
      body: BackgroundWrapper( // ✅ Apply fixed background
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Center(
  child: Stack(
    children: [
      // ✅ Circle Avatar with Transparent Background & White Border
      Container(
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white, width: 2), // ✅ Thin white border
        ),
        child: CircleAvatar(
          radius: 50,
          backgroundColor: Colors.transparent, // ✅ Fully transparent background
          backgroundImage: _imageBytes == null ? null : MemoryImage(_imageBytes!),
          child: _imageBytes == null
              ? Text(
                  nameController.text.isNotEmpty ? nameController.text[0] : 'U',
                  style: const TextStyle(fontSize: 40, color: Colors.white),
                )
              : null,
        ),
      ),
      
      // ✅ Edit Button (Bottom Right)
      Positioned(
        bottom: 0,
        right: 0,
        child: CircleAvatar(
          radius: 20,
          backgroundColor: Colors.white,
          child: IconButton(
            icon: const Icon(Icons.edit, size: 16, color: Colors.blueAccent),
            onPressed: _pickImage,
          ),
        ),
      ),
    ],
  ),
),

              const SizedBox(height: 20),
              _buildTextField(nameController, 'Full Name'),
              _buildTextField(emailController, 'Email'),
              _buildTextField(phoneController, 'Phone Number'),
              _buildTextField(passwordController, 'Password', obscureText: true),
              const SizedBox(height: 30),

              // ✅ Save Changes Button
              _buildSaveButton(),
            ],
          ),
        ),
      ),
    );
  }

  // ✅ Custom method to build Save Button
  Widget _buildSaveButton() {
    return Center(
      child: ConstrainedBox(
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
            onPressed: _saveUserProfile,
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
              'Save Changes',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTextField(TextEditingController controller, String labelText, {bool obscureText = false}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 20.0),
      child: TextField(
        controller: controller,
        obscureText: obscureText,
        decoration: InputDecoration(
          labelText: labelText,
          labelStyle: const TextStyle(color: Colors.white70),
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
}
