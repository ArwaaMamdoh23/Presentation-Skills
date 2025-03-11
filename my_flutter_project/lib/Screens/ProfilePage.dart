import 'package:flutter/material.dart';
import 'package:my_flutter_project/Screens/HomePage.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:typed_data';
import '../widgets/custom_app_bar.dart'; // ✅ Import Custom AppBar
import '../widgets/background_wrapper.dart'; // ✅ Import Background Wrapper
import 'EditProfilePage.dart';
// import 'SignOutPage.dart';
import 'SettingsPage.dart';
import 'package:my_flutter_project/AdminFolder/AdminDashboard.dart'; // ✅ Import Admin Dashboard
import '../widgets/CustomDrawer .dart'; 

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  _ProfilePageState createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  String? name;
  String? email;
  String? profession;
  Uint8List? _imageBytes;

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
      MaterialPageRoute(builder: (context) => const HomePage()),
    );
  }

  void _goToSettings() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const SettingsPage()),
    );
  }

  void _goToDashboard() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const AdminDashboard()), // ✅ Navigate to AdminDashboard
    );
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
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Center(
                  child: Stack(
                    children: [
                      // ✅ Profile Image with Transparent Background & White Border
                      Container(
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 2), // ✅ Thin white border
                        ),
                        child: CircleAvatar(
                          radius: 50,
                          backgroundColor: Colors.transparent, // ✅ Fully transparent
                          backgroundImage: _imageBytes == null ? null : MemoryImage(_imageBytes!),
                          child: _imageBytes == null
                              ? Text(
                                  name != null ? name![0] : 'U',
                                  style: const TextStyle(fontSize: 40, color: Colors.white),
                                )
                              : null,
                        ),
                      ),

                      // ✅ Edit Profile Image Button
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
                _buildInfoTile('Name', name),
                _buildInfoTile('Email', email),
                _buildInfoTile('Profession', profession),
                const SizedBox(height: 30),

                _buildButton('Edit Profile', () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => const EditProfilePage()),
                  );
                }),

                const SizedBox(height: 20),

                _buildButton('Dashboard', _goToDashboard), // ✅ Updated button for Dashboard

                const SizedBox(height: 30),

                Column(
                  children: [
                    _buildListTile(Icons.payment, 'Billing Details'),
                    _buildListTile(Icons.account_box, 'User Management'),
                    _buildListTile(Icons.info, 'Information'),
                    _buildListTile(Icons.settings, 'Settings', _goToSettings),
                    _buildListTile(Icons.exit_to_app, 'Logout', _signOut),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildInfoTile(String label, String? value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Text(
        '$label: ${value ?? "N/A"}',
        style: const TextStyle(fontSize: 18, color: Colors.white),
      ),
    );
  }

  Widget _buildButton(String text, VoidCallback onPressed) {
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
            onPressed: onPressed,
            style: ElevatedButton.styleFrom(
              minimumSize: const Size(280, 60),
              backgroundColor: Colors.transparent,
              shadowColor: Colors.transparent,
              padding: const EdgeInsets.symmetric(vertical: 20),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(30),
              ),
            ),
            child: Text(
              text,
              style: const TextStyle(
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

  Widget _buildListTile(IconData icon, String title, [VoidCallback? onTap]) {
    return ListTile(
      leading: Icon(icon, color: Colors.white),
      title: Text(title, style: const TextStyle(color: Colors.white)),
      onTap: onTap,
    );
  }
}