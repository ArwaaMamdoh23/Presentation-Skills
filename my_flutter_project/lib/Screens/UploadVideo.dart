import 'package:flutter/material.dart';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import '../widgets/custom_app_bar.dart'; // ✅ Use custom AppBar
// import 'SettingsPage.dart';
// import 'ProfilePage.dart';

class UploadVideoPage extends StatefulWidget {
  const UploadVideoPage({super.key});

  @override
  _UploadVideoPageState createState() => _UploadVideoPageState();
}

class _UploadVideoPageState extends State<UploadVideoPage> {
  File? _videoFile;

  Future<void> _pickVideo() async {
    try {
      final pickedFile = await ImagePicker().pickVideo(source: ImageSource.gallery);
      if (pickedFile != null) {
        setState(() {
          _videoFile = File(pickedFile.path);
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No video selected')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      extendBodyBehindAppBar: true, // ✅ Extends content behind the AppBar
      appBar: CustomAppBar(
      showSignIn: false, // User is signed in, so we hide the Sign-In button
      isUserSignedIn: true, // Ensures Profile & Settings icons appear
),

      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          // ✅ Background Image
          Container(
            decoration: const BoxDecoration(
              image: DecorationImage(
                image: AssetImage('assets/images/back.jpg'),
                fit: BoxFit.cover,
              ),
            ),
          ),
          Center(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 30.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  const Text(
                    'Upload Your Presentation',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 30),
                  
                  // ✅ Video Selection Box
                  Container(
                    padding: const EdgeInsets.all(10),
                    width: double.infinity,
                    height: 60,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(30),
                      border: Border.all(color: Colors.white),
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      _videoFile != null
                          ? _videoFile!.path.split('/').last
                          : 'No video selected', // ✅ Ensures proper display
                      style: const TextStyle(color: Colors.white),
                      textAlign: TextAlign.center,
                    ),
                  ),
                  
                  const SizedBox(height: 20),

                  // ✅ Upload Video Button
                  ElevatedButton(
                    onPressed: _pickVideo,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.transparent,
                      shadowColor: Colors.transparent,
                      padding: EdgeInsets.zero, // ✅ Removes default button padding
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    child: Ink(
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
                      ),
                      child: Container(
                        constraints: const BoxConstraints(minWidth: 200, minHeight: 50),
                        alignment: Alignment.center,
                        child: const Text(
                          'Upload Video',
                          style: TextStyle(fontSize: 18, color: Colors.white),
                        ),
                      ),
                    ),
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
