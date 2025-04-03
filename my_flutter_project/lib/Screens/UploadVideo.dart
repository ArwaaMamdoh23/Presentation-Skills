import 'package:flutter/material.dart';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import '../widgets/custom_app_bar.dart'; // ✅ Use custom AppBar
import '../widgets/background_wrapper.dart'; // Import BackgroundWrapper
import '../widgets/CustomDrawer .dart'; 
import 'package:firebase_storage/firebase_storage.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:path/path.dart' as path;
import 'SettingsPage.dart';
import 'ProfilePage.dart';

class UploadVideoPage extends StatefulWidget {
  const UploadVideoPage({super.key});

  @override
  _UploadVideoPageState createState() => _UploadVideoPageState();
}

class _UploadVideoPageState extends State<UploadVideoPage> {
  File? _videoFile;
  bool isSignedIn = true;  // Simulate user sign-in status
  bool _isUploading = false;
  String? _downloadURL;
  Future<void> _pickVideo() async {
    try {
      final pickedFile = await ImagePicker().pickVideo(source: ImageSource.gallery);
      if (pickedFile != null) {
        setState(() {
          _videoFile = File(pickedFile.path);
        });

         // ✅ Navigate to Reports Page After Upload
      Future.delayed(const Duration(seconds: 1), () {  // ✅ Delay for better UX
        Navigator.pushReplacementNamed(context, '/report');  // ✅ Change route name if needed
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

  Future<void> _uploadVideo() async {
    if (_videoFile == null) return;

    setState(() {
      _isUploading = true;
    });

    try {
      String fileName = path.basename(_videoFile!.path);
      Reference storageRef = FirebaseStorage.instance.ref().child('uploads/$fileName');

      UploadTask uploadTask = storageRef.putFile(_videoFile!);
      TaskSnapshot snapshot = await uploadTask;

      String downloadURL = await snapshot.ref.getDownloadURL();

      // Save video URL in Firestore temporarily
      await FirebaseFirestore.instance.collection('videos').add({
        'url': downloadURL,
        'timestamp': FieldValue.serverTimestamp(),
        'processed': false, // Indicates it hasn't been processed yet
      });

      setState(() {
        _downloadURL = downloadURL;
        _isUploading = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Upload successful! Video saved temporarily.')),
      );
    } catch (e) {
      setState(() {
        _isUploading = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Upload failed: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true, // ✅ Extends content behind the AppBar
      appBar: CustomAppBar(
        showSignIn: false, // User is signed in, so we hide the Sign-In button
        isUserSignedIn: isSignedIn, // Ensures Profile & Settings icons appear
      ),
      backgroundColor: Colors.transparent,
      drawer: CustomDrawer(isSignedIn: isSignedIn),  // Add the custom drawer here
      body: BackgroundWrapper(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 30.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                const Text(
                  'Upload Your Presentation',
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
      ),
    );
  }
}
