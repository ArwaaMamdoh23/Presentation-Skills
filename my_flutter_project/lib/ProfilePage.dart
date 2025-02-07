import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:file_picker/file_picker.dart';  // Import file_picker for Flutter Web
import 'dart:typed_data'; // For handling image bytes
import 'dart:ui'; // Import for ImageFilter
import 'EditProfilePage.dart';  // Import EditProfilePage
import 'SignOutPage.dart';  // Import SignOutPage

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  _ProfilePageState createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  String? name;
  String? email;
  String? profession;
  Uint8List? _imageBytes; // Store image as bytes

  @override
  void initState() {
    super.initState();
    _loadUserProfile();
  }

  void _loadUserProfile() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      name = prefs.getString('name') ?? 'Arwaa Mamdoh';
      email = prefs.getString('email') ?? 'arwaa2110478@miuegypt.edu.eg';
      profession = prefs.getString('profession') ?? 'Student';
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

  void _signOut() {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => 
       SignOutPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage('assets/images/back.jpg'),
                fit: BoxFit.cover,
                colorFilter: ColorFilter.mode(
                  Colors.black.withOpacity(0.4),
                  BlendMode.darken,
                ),
              ),
            ),
            child: ClipRect(
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 5, sigmaY: 5),
                child: Container(
                  color: Colors.transparent,
                ),
              ),
            ),
          ),
          SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Center(
                    child: Stack(
                      children: [
                        CircleAvatar(
                          radius: 50,
                          backgroundColor: Colors.blueAccent,
                          backgroundImage: _imageBytes == null ? null : MemoryImage(_imageBytes!),
                          child: _imageBytes == null
                              ? Text(
                                  name != null ? name![0] : 'U',
                                  style: const TextStyle(fontSize: 40, color: Colors.white),
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
                              icon: const Icon(Icons.edit, size: 16, color: Colors.blueAccent),
                              onPressed: _pickImage,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 20),
                  Text('Name: $name', style: const TextStyle(fontSize: 18, color: Colors.white)),
                  Text('Email: $email', style: const TextStyle(fontSize: 18, color: Colors.white)),
                  Text('Profession: $profession', style: const TextStyle(fontSize: 18, color: Colors.white)),
                  const SizedBox(height: 30),

                  Center(
                    child: ConstrainedBox(
                      constraints: BoxConstraints(maxWidth: 280),
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
                              offset: Offset(0, 5),
                            ),
                          ],
                        ),
                        child: ElevatedButton(
                          onPressed: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(builder: (context) => const EditProfilePage()),
                            );
                          },
                          style: ElevatedButton.styleFrom(
                            minimumSize: Size(280, 60),
                            backgroundColor: Colors.transparent,
                            shadowColor: Colors.transparent,
                            padding: const EdgeInsets.symmetric(vertical: 20),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                          ),
                          child: const Text(
                            'Edit Profile',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 30),

                  const ListTile(
                    leading: Icon(Icons.settings, color: Colors.white),
                    title: Text('Settings', style: TextStyle(color: Colors.white)),
                  ),
                  const ListTile(
                    leading: Icon(Icons.payment, color: Colors.white),
                    title: Text('Billing Details', style: TextStyle(color: Colors.white)),
                  ),
                  const ListTile(
                    leading: Icon(Icons.account_box, color: Colors.white),
                    title: Text('User Management', style: TextStyle(color: Colors.white)),
                  ),
                  const ListTile(
                    leading: Icon(Icons.info, color: Colors.white),
                    title: Text('Information', style: TextStyle(color: Colors.white)),
                  ),
                  ListTile(
                    leading: Icon(Icons.exit_to_app, color: Colors.white),
                    title: Text('Logout', style: TextStyle(color: Colors.white)),
                    onTap: _signOut, // Redirect to SignOutPage when tapped
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
