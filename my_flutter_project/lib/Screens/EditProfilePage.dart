import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:typed_data';
import 'dart:convert';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:easy_localization/easy_localization.dart';  // Import easy_localization
import '../widgets/custom_app_bar.dart';
import '../widgets/background_wrapper.dart';

class EditProfilePage extends StatefulWidget {
  const EditProfilePage({super.key});

  @override
  State<EditProfilePage> createState() => _EditProfilePageState();
}

class _EditProfilePageState extends State<EditProfilePage> {
  final _supabase = Supabase.instance.client;
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _currentPasswordController = TextEditingController();
  final TextEditingController _newPasswordController = TextEditingController();
  final TextEditingController _confirmPasswordController = TextEditingController();
  final _formKey = GlobalKey<FormState>();

  Uint8List? _profileImageBytes;
  bool _isLoading = false;

  bool hasUppercase = false;
  bool hasNumber = false;
  bool hasSpecialChar = false;
  bool hasMinLength = false;
  bool showPasswordRequirements = false;
  bool passwordsMatch = true;

  @override
  void initState() {
    super.initState();
    _loadUserProfile();
  }

  Future<void> _loadUserProfile() async {
    setState(() => _isLoading = true);
    final user = _supabase.auth.currentUser;
    if (user == null || !mounted) return;

    try {
      final response = await _supabase
          .from('User')
          .select()
          .eq('User_id', user.id)
          .single();

      if (mounted) {
        _nameController.text = response['Name'] ?? '';
        final imageString = response['ProfileImage'] as String?;
        if (imageString != null && imageString.isNotEmpty) {
          setState(() => _profileImageBytes = base64Decode(imageString));
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to load profile: ${e.toString()}'.tr())),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _validatePassword(String password) {
    setState(() {
      hasUppercase = RegExp(r'[A-Z]').hasMatch(password);
      hasNumber = RegExp(r'[0-9]').hasMatch(password);
      hasSpecialChar = RegExp(r'[!@#$%^&*(),.?":{}|<>]').hasMatch(password);
      hasMinLength = password.length >= 8;
      showPasswordRequirements = password.isNotEmpty;
      _validatePasswordMatch();
    });
  }

  void _validatePasswordMatch() {
    setState(() {
      passwordsMatch = _newPasswordController.text == _confirmPasswordController.text;
    });
  }

  Future<void> _pickImage() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        allowMultiple: false,
      );

      if (result != null && result.files.isNotEmpty && mounted) {
        setState(() => _profileImageBytes = result.files.first.bytes);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to pick image: ${e.toString()}'.tr())),
        );
      }
    }
  }

  Future<void> _saveProfile() async {
    if (!mounted) return;
    setState(() => _isLoading = true);

    final user = _supabase.auth.currentUser;
    if (user == null || user.email == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
           SnackBar(content: Text('No authenticated user found'.tr())),
        );
      }
      return;
    }

    final newName = _nameController.text.trim();
    final newPassword = _newPasswordController.text.trim();
    final currentPassword = _currentPasswordController.text.trim();
    final base64Image = _profileImageBytes != null ? base64Encode(_profileImageBytes!) : null;

    if (newName.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Name cannot be empty'.tr())),
      );
      return;
    }

    try {
      await _supabase
          .from('User')
          .update({
            'Name': newName,
            if (base64Image != null) 'ProfileImage': base64Image,
          })
          .eq('User_id', user.id);

      if (newPassword.isNotEmpty) {
        final authResponse = await _supabase.auth.signInWithPassword(
          email: user.email!,
          password: currentPassword,
        );

        if (authResponse.user == null) {
          throw Exception('Reauthentication failed');
        }

        await _supabase.auth.updateUser(UserAttributes(password: newPassword));
      }

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Profile updated successfully!'.tr())),
        );
        Navigator.pop(context);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to update profile: ${e.toString()}'.tr())),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Widget _buildTextField(TextEditingController controller, String hintText,
      {bool obscureText = false, String? Function(String?)? validator, void Function(String)? onChanged}) {
    return TextFormField(
      controller: controller,
      obscureText: obscureText,
      onChanged: onChanged,
      decoration: InputDecoration(
        hintText: hintText.tr(), // Use tr() for translation
        hintStyle: const TextStyle(color: Colors.white70),
        filled: true,
        fillColor: Colors.white.withOpacity(0.2),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(30),
          borderSide: BorderSide.none,
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
      ),
      style: const TextStyle(color: Colors.white),
      validator: validator,
    );
  }

  Widget _buildPasswordRequirement(String text, bool isMet) {
    return Row(
      children: [
        Icon(Icons.check_circle, color: isMet ? Colors.green : Colors.grey),
        const SizedBox(width: 5),
        Text(text.tr(), style: TextStyle(color: isMet ? Colors.green : Colors.white70)),
      ],
    );
  }

  Widget _buildPasswordRequirements() {
    return Container(
      margin: const EdgeInsets.only(top: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.8),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("Password Requirements".tr(), style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
          const SizedBox(height: 5),
          _buildPasswordRequirement('At least one uppercase letter'.tr(), hasUppercase),
          _buildPasswordRequirement('At least one number'.tr(), hasNumber),
          _buildPasswordRequirement('At least one special character'.tr(), hasSpecialChar),
          _buildPasswordRequirement('Minimum 8 characters'.tr(), hasMinLength),
          _buildPasswordRequirement('Passwords match'.tr(), passwordsMatch),
        ],
      ),
    );
  }

  Widget _buildSaveButton() {
    return ConstrainedBox(
      constraints: const BoxConstraints(maxWidth: 280),
      child: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.blueGrey.shade900, Colors.blueGrey.shade700],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(30),
          boxShadow: [
            BoxShadow(color: Colors.lightBlue.withOpacity(0.4), blurRadius: 15, offset: const Offset(0, 5)),
          ],
        ),
        child: ElevatedButton(
          onPressed: _isLoading ? null : _saveProfile,
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.transparent,
            shadowColor: Colors.transparent,
            minimumSize: const Size(280, 60),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
          ),
          child: _isLoading
              ? const CircularProgressIndicator(color: Colors.white)
              : Text('Save Changes'.tr(), style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: CustomAppBar(showSignIn: false, isUserSignedIn: true),
      body: BackgroundWrapper(
        child: _isLoading
            ? const Center(child: CircularProgressIndicator())
            : SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Form(
                  key: _formKey,
                  child: Column(
                    children: [
                      const SizedBox(height: kToolbarHeight + 20),
                      Center(
                        child: Stack(
                          children: [
                            CircleAvatar(
                              radius: 60,
                              backgroundColor: Colors.grey[800],
                              backgroundImage: _profileImageBytes != null
                                  ? MemoryImage(_profileImageBytes!)
                                  : null,
                              child: _profileImageBytes == null
                                  ? const Icon(Icons.person, size: 60, color: Colors.white)
                                  : null,
                            ),
                            Positioned(
                              bottom: 0,
                              right: 0,
                              child: GestureDetector(
                                onTap: _pickImage,
                                child: Container(
                                  padding: const EdgeInsets.all(8),
                                  decoration: const BoxDecoration(
                                    color: Colors.blue,
                                    shape: BoxShape.circle,
                                  ),
                                  child: const Icon(Icons.edit, size: 20, color: Colors.white),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 24),
                      _buildTextField(_nameController, 'Full Name'.tr(), validator: (value) {
                        if (value == null || value.isEmpty) return 'Name cannot be empty'.tr();
                        return null;
                      }),
                      const SizedBox(height: 16),
                      _buildTextField(_currentPasswordController, 'Current Password'.tr(), obscureText: true),
                      const SizedBox(height: 16),
                      _buildTextField(_newPasswordController, 'New Password'.tr(), obscureText: true, onChanged: _validatePassword),
                      const SizedBox(height: 16),
                      _buildTextField(_confirmPasswordController, 'Confirm New Password'.tr(), obscureText: true, validator: (value) {
                        if (value != _newPasswordController.text) return 'Passwords do not match'.tr();
                        return null;
                      }),
                      if (showPasswordRequirements) _buildPasswordRequirements(),
                      const SizedBox(height: 30),
                      _buildSaveButton(),
                    ],
                  ),
                ),
              ),
      ),
    );
  }
}
