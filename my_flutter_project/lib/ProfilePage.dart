import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:file_picker/file_picker.dart';  // Import file_picker instead of image_picker
import 'dart:io'; // For handling files
import 'EditProfilePage.dart';  // Import EditProfilePage

class ProfilePage extends StatefulWidget {
  @override
  _ProfilePageState createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  String? name;
  String? email;
  String? profession;
  File? _image; // This will hold the selected image

  @override
  void initState() {
    super.initState();
    _loadUserProfile();
  }

  // Load user profile data from SharedPreferences
  void _loadUserProfile() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      name = prefs.getString('name') ?? 'User';
      email = prefs.getString('email') ?? 'user@example.com';
      profession = prefs.getString('profession') ?? 'Student';
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Profile'),
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
                            name != null ? name![0] : 'U', // Display initials if no image is selected
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
            Text('Name: $name', style: TextStyle(fontSize: 18, color: Colors.black)),
            Text('Email: $email', style: TextStyle(fontSize: 18, color: Colors.black)),
            Text('Profession: $profession', style: TextStyle(fontSize: 18, color: Colors.black)),
            SizedBox(height: 30),

            // Edit Profile Button (Navigates to Edit Profile Page)
            Center(
              child: ElevatedButton(
                onPressed: () {
                  // Navigate to the Edit Profile page
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => EditProfilePage()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blueAccent,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  padding: EdgeInsets.symmetric(vertical: 15),
                  minimumSize: Size(200, 50),
                ),
                child: Text('Edit Profile', style: TextStyle(fontSize: 18, color: Colors.white)),
              ),
            ),
            SizedBox(height: 30),

            // Additional Options Section
            ListTile(
              leading: Icon(Icons.settings, color: Colors.blueAccent),
              title: Text('Settings', style: TextStyle(color: Colors.black)),
            ),
            ListTile(
              leading: Icon(Icons.payment, color: Colors.blueAccent),
              title: Text('Billing Details', style: TextStyle(color: Colors.black)),
            ),
            ListTile(
              leading: Icon(Icons.account_box, color: Colors.blueAccent),
              title: Text('User Management', style: TextStyle(color: Colors.black)),
            ),
            ListTile(
              leading: Icon(Icons.info, color: Colors.blueAccent),
              title: Text('Information', style: TextStyle(color: Colors.black)),
            ),
            ListTile(
              leading: Icon(Icons.exit_to_app, color: Colors.blueAccent),
              title: Text('Logout', style: TextStyle(color: Colors.black)),
            ),
          ],
        ),
      ),
    );
  }
}
