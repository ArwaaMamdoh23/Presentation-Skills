import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart'; // For picking videos from the gallery
import 'package:supabase_flutter/supabase_flutter.dart'; // For interacting with Supabase
import 'package:path/path.dart' as path; // For handling file paths
import 'package:file_picker/file_picker.dart'; // For picking files from the system
import 'package:googleapis/drive/v3.dart' as drive; // For interacting with Google Drive API
import 'package:googleapis_auth/googleapis_auth.dart'; // For OAuth2 authentication
import 'package:google_sign_in/google_sign_in.dart'; // For Google Sign-In functionality
import 'package:http/http.dart' as http; // For making HTTP requests
import 'package:flutter/services.dart' show rootBundle; // For loading assets
import 'dart:convert'; // For JSON handling

import '../widgets/background_wrapper.dart'; // Custom widget for background styling
import 'EditProfilePage.dart'; // EditProfilePage widget
import 'ProfilePage.dart'; // ProfilePage widget
import '../widgets/custom_app_bar.dart'; // Custom AppBar widget
import '../widgets/CustomDrawer .dart'; // Custom Drawer widget

class UploadVideoPage extends StatefulWidget {
  @override
  _UploadVideoPageState createState() => _UploadVideoPageState();
}

class _UploadVideoPageState extends State<UploadVideoPage> {
  File? _videoFile;
  bool _isUploading = false;
  double _uploadProgress = 0;
  final _supabase = Supabase.instance.client;
  final _picker = ImagePicker();
  final _googleSignIn = GoogleSignIn(scopes: ['https://www.googleapis.com/auth/drive.file']);

  // Function to pick video from the gallery
  Future<void> _pickVideoFromGallery() async {
    try {
      final pickedFile = await _picker.pickVideo(
        source: ImageSource.gallery,
        maxDuration: const Duration(minutes: 5),
      );

      if (pickedFile != null) {
        setState(() {
          _videoFile = File(pickedFile.path);
          _uploadProgress = 0;
        });
      }
    } catch (e) {
      _showError('Error selecting video: ${e.toString()}');
    }
  }

  // Function to pick video from Google Drive
  Future<void> _pickVideoFromDrive() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.video,
        allowMultiple: false,
      );

      if (result != null) {
        setState(() {
          _videoFile = File(result.files.single.path!);
          _uploadProgress = 0;
        });
      }
    } catch (e) {
      _showError('Error selecting video: ${e.toString()}');
    }
  }

  // Function to load credentials for Google API authentication
  Future<Map<String, dynamic>> loadCredentials() async {
    final String credentialsJson = await rootBundle.loadString('assets/credentials.json');
    return json.decode(credentialsJson);
  }

  // Function to authenticate with Google via OAuth2
  Future<void> _authenticateWithGoogle() async {
    try {
      // Use GoogleSignIn to authenticate the user
      final GoogleSignInAccount? account = await _googleSignIn.signIn();

      if (account == null) {
        _showError('Google Sign-In failed.');
        return;
      }

      final GoogleSignInAuthentication auth = await account.authentication;

      // Get the authentication token (access token)
      final String accessToken = auth.accessToken!;

      // Create the authenticated HTTP client
      final authHeaders = {'Authorization': 'Bearer $accessToken'};
      final authenticateClient = http.Client();

      // Use the authenticated client to interact with Google APIs
      final driveApi = drive.DriveApi(authenticateClient);

      // You can now interact with Google Drive
      print("Authenticated successfully!");

    } catch (e) {
      print('Authentication failed: $e');
      _showError('Authentication failed: $e');
    }
  }

  // Function to upload the selected video to Google Drive
  Future<void> _uploadVideo() async {
    if (_videoFile == null) {
      _showError('Please select a video first');
      return;
    }

    if (!mounted) return;
    setState(() {
      _isUploading = true;
      _uploadProgress = 0;
    });

    try {
      final account = _googleSignIn.currentUser;
      if (account == null) {
        await _authenticateWithGoogle();
      }

      final authHeaders = await account!.authHeaders;
      final authenticateClient = GoogleHttpClient(authHeaders);

      final driveApi = drive.DriveApi(authenticateClient);

      final media = drive.Media(_videoFile!.openRead(), _videoFile!.lengthSync());
      final driveFile = drive.File()
        ..name = 'video_${DateTime.now().millisecondsSinceEpoch}.mp4'
        ..mimeType = 'video/mp4';

      final response = await driveApi.files.create(
        driveFile,
        uploadMedia: media,
      );

      final fileUrl = 'https://drive.google.com/file/d/${response.id}/view';
      _showSuccess('Video uploaded successfully! URL: $fileUrl');
    } catch (e) {
      _showError('Upload failed: ${e.toString()}');
    } finally {
      if (mounted) {
        setState(() => _isUploading = false);
      }
    }
  }

  // Helper method to get the mime type for different video extensions
  String _getMimeType(String extension) {
    switch (extension) {
      case '.mp4':
        return 'video/mp4';
      case '.mov':
        return 'video/quicktime';
      case '.avi':
        return 'video/x-msvideo';
      case '.mkv':
        return 'video/x-matroska';
      case '.webm':
        return 'video/webm';
      default:
        return 'video/mp4';
    }
  }

  // Helper method to show error messages
  void _showError(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.red,
          duration: const Duration(seconds: 3),
        ),
      );
    }
  }

  // Helper method to show success messages
  void _showSuccess(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.green,
          duration: const Duration(seconds: 2),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: CustomAppBar(
        showSignIn: true,
        isUserSignedIn: true,
      ),
      drawer: CustomDrawer(isSignedIn: true),
      body: BackgroundWrapper(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                'Upload Presentation',
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
              const SizedBox(height: 20),
              if (_videoFile != null) ...[
                const Icon(Icons.video_library, size: 60),
                const SizedBox(height: 10),
                Text(
                  path.basename(_videoFile!.path),
                  style: Theme.of(context).textTheme.titleMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
              ],
              if (_isUploading) ...[
                LinearProgressIndicator(value: _uploadProgress),
                const SizedBox(height: 15),
                Text(
                  'Uploading: ${(_uploadProgress * 100).toStringAsFixed(1)}%',
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
                const SizedBox(height: 30),
              ],
              ElevatedButton.icon(
                onPressed: _pickVideoFromGallery,
                icon: const Icon(Icons.photo_library),
                label: const Text('Select from Gallery'),
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 50),
                  backgroundColor: Colors.transparent,
                  shadowColor: Colors.transparent,
                ),
              ),
              ElevatedButton.icon(
                onPressed: _pickVideoFromDrive,
                icon: const Icon(Icons.folder),
                label: const Text('Select from Drive'),
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 50),
                  backgroundColor: Colors.transparent,
                  shadowColor: Colors.transparent,
                ),
              ),
              ElevatedButton.icon(
                onPressed: _isUploading ? null : _uploadVideo,
                icon: _isUploading
                    ? const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.cloud_upload),
                label: const Text('Upload Video'),
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 50),
                  backgroundColor: _isUploading
                      ? Colors.transparent
                      : Colors.transparent,
                  shadowColor: Colors.transparent,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// GoogleHttpClient class to handle HTTP requests
class GoogleHttpClient extends http.BaseClient {
  final Map<String, String> _headers;
  GoogleHttpClient(this._headers);

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    request.headers.addAll(_headers);
    final streamedResponse = await http.Client().send(request);
    return streamedResponse;
  }
}
