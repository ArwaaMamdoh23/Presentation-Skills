import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:file_picker/file_picker.dart'; // File picker package
import 'dart:io'; // For handling files

class EditProfilePage extends StatefulWidget {
  @override
  _EditProfilePageState createState() => _EditProfilePageState();
}

class _EditProfilePageState extends State<EditProfilePage> {
  TextEditingController nameController = TextEditingController();
  TextEditingController emailController = TextEditingController();
  TextEditingController phoneController = TextEditingController();
  TextEditingController passwordController = TextEditingController();
  File? _image; // This will hold the selected image

  @override
  void initState() {
    super.initState();
    _loadUserProfile(); // Load the profile data from SharedPreferences
  }

  // Load user profile data from SharedPreferences
  void _loadUserProfile() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      nameController.text = prefs.getString('name') ?? '';
      emailController.text = prefs.getString('email') ?? '';
      phoneController.text = prefs.getString('phone') ?? '';
      passwordController.text = prefs.getString('password') ?? '';
      String? imagePath = prefs.getString('profile_image');
      if (imagePath != null) {
        _image = File(imagePath); // Load saved image if available
      }
    });
  }

  // Select an image from the gallery using file_picker (for web)
  Future<void> _pickImage() async {
    // Use file_picker for web environments
    FilePickerResult? result = await FilePicker.platform.pickFiles(type: FileType.image);

    if (result != null) {
      // Get the file path
      PlatformFile file = result.files.first;
      setState(() {
        _image = File(file.path!); // Convert the picked file into a File object
      });

      // Save the image path to SharedPreferences
      SharedPreferences prefs = await SharedPreferences.getInstance();
      prefs.setString('profile_image', file.path!); // Save image path
    }
  }

  // Save user profile data to SharedPreferences
  void _saveUserProfile() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setString('name', nameController.text);
    prefs.setString('email', emailController.text);
    prefs.setString('phone', phoneController.text);
    prefs.setString('password', passwordController.text);
    if (_image != null) {
      prefs.setString('profile_image', _image!.path); // Save the selected profile image
    }
    Navigator.pop(context); // Go back to ProfilePage after saving
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Edit Profile'),
        backgroundColor: Color.fromARGB(255, 169, 171, 172), // Grey background color
      ),
      body: Container(
        color: Color.fromARGB(255, 195, 213, 226), // Light grey background
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            // Profile Picture with Edit Icon
            Center(
              child: Stack(
                children: [
                  CircleAvatar(
                    radius: 50,
                    backgroundColor: Colors.blueAccent,
                    backgroundImage: _image == null ? null : FileImage(_image!), // Show the selected image if available
                    child: _image == null
                        ? Text(
                            nameController.text.isNotEmpty ? nameController.text[0] : 'U', // Show initial if no image
                            style: TextStyle(fontSize: 40, color: Colors.white),
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
                        icon: Icon(Icons.edit, size: 16, color: Colors.blueAccent),
                        onPressed: _pickImage, // Trigger the image picker when the icon is pressed
                      ),
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 20),
            TextField(
              controller: nameController,
              decoration: InputDecoration(labelText: 'Full Name', labelStyle: TextStyle(color: Colors.black)),
            ),
            TextField(
              controller: emailController,
              decoration: InputDecoration(labelText: 'Email', labelStyle: TextStyle(color: Colors.black)),
            ),
            TextField(
              controller: phoneController,
              decoration: InputDecoration(labelText: 'Phone Number', labelStyle: TextStyle(color: Colors.black)),
            ),
            TextField(
              controller: passwordController,
              obscureText: true,
              decoration: InputDecoration(labelText: 'Password', labelStyle: TextStyle(color: Colors.black)),
            ),
            SizedBox(height: 30),
            ElevatedButton(
              onPressed: _saveUserProfile,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blueAccent,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
                padding: EdgeInsets.symmetric(vertical: 15),
                minimumSize: Size(200, 50),
              ),
              child: Text('Save Changes', style: TextStyle(fontSize: 18, color: Colors.white)),
            ),
          ],
        ),
      ),
    );
  }
}
