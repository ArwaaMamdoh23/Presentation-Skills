import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:path/path.dart' as path;

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

  Future<void> _pickVideo() async {
    try {
      final pickedFile = await _picker.pickVideo(
        source: ImageSource.gallery,
        maxDuration: const Duration(minutes: 10),
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
      final userId = _supabase.auth.currentUser?.id;
      if (userId == null) throw Exception('User not authenticated');

      final fileExt = path.extension(_videoFile!.path).toLowerCase();
      final fileName = 'video_${DateTime.now().millisecondsSinceEpoch}$fileExt';
      final filePath = 'user_uploads/$userId/$fileName';

      // Upload to storage
      await _supabase.storage
          .from('videos')
          .upload(
            filePath, 
            _videoFile!,
            fileOptions: FileOptions(
              contentType: _getMimeType(fileExt),
            ),
          );

      // Get public URL
      final fileUrl = _supabase.storage
          .from('videos')
          .getPublicUrl(filePath);

      // Insert metadata
      await _supabase.from('Uploaded_file').insert({
        'User_id': userId,
        'File_name': fileName,
        'File_path': fileUrl,
        'File_type': 'video${fileExt.replaceFirst('.', '')}',
      });

      _showSuccess('Video uploaded successfully!');
      if (mounted) {
        Navigator.of(context).pop();
      }
    } catch (e) {
      _showError('Upload failed: ${e.toString()}');
    } finally {
      if (mounted) {
        setState(() => _isUploading = false);
      }
    }
  }

  String _getMimeType(String extension) {
    switch (extension) {
      case '.mp4': return 'video/mp4';
      case '.mov': return 'video/quicktime';
      case '.avi': return 'video/x-msvideo';
      default: return 'video/mp4';
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
      appBar: AppBar(
        title: const Text('Upload Video'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
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
              onPressed: _pickVideo,
              icon: const Icon(Icons.video_library),
              label: const Text('Select Video'),
              style: ElevatedButton.styleFrom(
                minimumSize: const Size(double.infinity, 50),
              ),
            ),
            const SizedBox(height: 20),
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
                backgroundColor: _isUploading ? Colors.grey : null,
              ),
            ),
          ],
        ),
      ),
    );
  }
}