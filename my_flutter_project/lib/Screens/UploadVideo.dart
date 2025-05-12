import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:my_flutter_project/Screens/Loading.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:path/path.dart' as path;
import 'package:file_picker/file_picker.dart';
import 'package:googleapis/drive/v3.dart' as drive;
import 'package:googleapis_auth/googleapis_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart' show rootBundle;
import 'package:easy_localization/easy_localization.dart';

import '../widgets/LanguageSwitcherIcon.dart';
import '../widgets/background_wrapper.dart';
import 'EditProfilePage.dart';
import 'ProfilePage.dart';
import '../widgets/custom_app_bar.dart';
import '../widgets/CustomDrawer .dart';

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
      _showError('Error selecting video: ${e.toString()}'.tr());
    }
  }

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
      _showError('Error selecting video: ${e.toString()}'.tr());
    }
  }

  Future<Map<String, dynamic>> loadCredentials() async {
    final String credentialsJson = await rootBundle.loadString('assets/credentials.json');
    return json.decode(credentialsJson);
  }

  Future<void> _authenticateWithGoogle() async {
    try {
      final GoogleSignInAccount? account = await _googleSignIn.signIn();

      if (account == null) {
        _showError('Google Sign-In failed.'.tr());
        return;
      }

      final GoogleSignInAuthentication auth = await account.authentication;
      final String accessToken = auth.accessToken!;
      final authHeaders = {'Authorization': 'Bearer $accessToken'};
      final authenticateClient = http.Client();

      final driveApi = drive.DriveApi(authenticateClient);
      print("Authenticated successfully!");
    } catch (e) {
      print('Authentication failed: $e');
      _showError('Authentication failed: $e'.tr());
    }
  }

  Future<void> _uploadVideo() async {
    if (_videoFile == null) {
      _showError('Please select a video first'.tr());
      return;
    }

    // Navigate to loading screen
    if (mounted) {
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => const LoadingScreen()),
      );
    }

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
      
      // Pop the loading screen and show success message
      if (mounted) {
        Navigator.pop(context);
        _showSuccess('Video uploaded successfully! URL: $fileUrl'.tr());
      }
    } catch (e) {
      // Pop the loading screen and show error message
      if (mounted) {
        Navigator.pop(context);
        _showError('Upload failed: ${e.toString()}'.tr());
      }
    }
  }

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
        extraActions: const [
          Padding(
            padding: EdgeInsets.only(right: 8),
            child: LanguageSwitcherIcon(),
          ),
        ],
      ),
      drawer: CustomDrawer(isSignedIn: true),
      body: BackgroundWrapper(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                'Upload Presentation'.tr(),
                style: const TextStyle(
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
                label:  Text('Select from Gallery'.tr()),
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 50),
                  backgroundColor: Colors.transparent,
                  shadowColor: Colors.transparent,
                ),
              ),
              ElevatedButton.icon(
                onPressed: _pickVideoFromDrive,
                icon: const Icon(Icons.folder),
                label:  Text('Select from Drive'.tr()),
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
                label:  Text('Upload Video'.tr()),
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 50),
                  backgroundColor: _isUploading ? Colors.transparent : Colors.transparent,
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
