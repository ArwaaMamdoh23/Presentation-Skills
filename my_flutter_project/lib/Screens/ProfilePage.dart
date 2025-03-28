import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:my_flutter_project/Screens/HomePage.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:typed_data';
import 'dart:convert'; // For base64 encoding/decoding
import 'package:connectivity_plus/connectivity_plus.dart'; // For internet check

import '../widgets/custom_app_bar.dart';
import '../widgets/background_wrapper.dart';
import 'EditProfilePage.dart';
import 'SettingsPage.dart';
import 'package:my_flutter_project/AdminFolder/AdminDashboard.dart';

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
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadUserProfile();
    _loadImage();
  }

  /// ✅ Check if there's an internet connection
  Future<bool> hasInternet() async {
    var result = await Connectivity().checkConnectivity();
    return result != ConnectivityResult.none;
  }

  /// ✅ Load user data from Firestore
  Future<void> _loadUserProfile() async {
    User? user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => const HomePage()),
      );
      return;
    }

    String userId = user.uid;

    if (!await hasInternet()) {
      print("⚠️ No internet connection.");
      setState(() => isLoading = false);
      return;
    }

    try {
      DocumentSnapshot userDoc =
          await FirebaseFirestore.instance.collection('User').doc(userId).get();

      print("📌 Firestore Data: ${userDoc.data()}");

      if (userDoc.exists && mounted) {
        setState(() {
          name = userDoc['Name'] ?? "N/A";
          email = userDoc['Email'] ?? user.email;
          profession = userDoc['Role'] ?? "N/A";
        });
      } else {
        print("⚠️ User document not found in Firestore!");
      }
    } catch (e) {
      print("❌ Firestore Error: $e");
    } finally {
      setState(() => isLoading = false);
    }
  }

  /// ✅ Load profile image from SharedPreferences
  Future<void> _loadImage() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String? storedImage = prefs.getString('profile_image_bytes');

    if (storedImage != null) {
      setState(() {
        _imageBytes = base64Decode(storedImage);
      });
    }
  }

  /// ✅ Sign out and go to HomePage
  void _signOut() async {
    await FirebaseAuth.instance.signOut();
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
      MaterialPageRoute(builder: (context) => const AdminDashboard()),
    );
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
        child: isLoading
            ? const Center(child: CircularProgressIndicator())
            : SingleChildScrollView(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: <Widget>[
                      Center(
                        child: Stack(
                          children: [
                            Container(
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                border: Border.all(color: Colors.white, width: 2),
                              ),
                              child: CircleAvatar(
                                radius: 50,
                                backgroundColor: Colors.transparent,
                                backgroundImage: _imageBytes == null
                                    ? null
                                    : MemoryImage(_imageBytes!),
                                child: _imageBytes == null
                                    ? Text(
                                        name != null ? name![0] : 'U',
                                        style: const TextStyle(
                                            fontSize: 40, color: Colors.white),
                                      )
                                    : null,
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
                          MaterialPageRoute(
                              builder: (context) => const EditProfilePage()),
                        );
                      }),

                      const SizedBox(height: 20),

                      _buildButton('Dashboard', _goToDashboard),

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
        child: ElevatedButton(
          onPressed: onPressed,
          style: ElevatedButton.styleFrom(
            minimumSize: const Size(280, 60),
            backgroundColor: Colors.blueGrey.shade700,
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
