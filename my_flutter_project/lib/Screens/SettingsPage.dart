import 'package:flutter/material.dart';
import 'dart:ui';

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background Image with Blur Effect
          Container(
            decoration: BoxDecoration(
              image: DecorationImage(
                image: const AssetImage('assets/images/back.jpg'),
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
          Column(
            children: [
              AppBar(
                title: const Text("Settings"),
                backgroundColor: Colors.transparent,
                elevation: 0,
              ),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      _buildSettingsItem(
                        icon: Icons.dark_mode,
                        title: "Theme",
                        subtitle: "Switch between Light and Dark mode",
                        onTap: () {
                          // Add theme switching logic here
                        },
                      ),
                      _buildSettingsItem(
                        icon: Icons.notifications,
                        title: "Notifications",
                        subtitle: "Manage notification preferences",
                        onTap: () {
                          // Navigate to notification settings
                        },
                      ),
                      _buildSettingsItem(
                        icon: Icons.privacy_tip,
                        title: "Terms & Privacy",
                        subtitle: "View our terms and privacy policy",
                        onTap: () {
                          // Navigate to Terms & Privacy page
                        },
                      ),
                      _buildSettingsItem(
                        icon: Icons.delete_forever,
                        title: "Delete Account",
                        subtitle: "Permanently remove your account",
                        onTap: () {
                          // Implement account deletion confirmation
                        },
                      ),
                      _buildSettingsItem(
                        icon: Icons.logout,
                        title: "Sign Out",
                        subtitle: "Log out from your account",
                        onTap: () {
                          // Implement sign-out functionality
                        },
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSettingsItem({required IconData icon, required String title, required String subtitle, required VoidCallback onTap}) {
    return Card(
      color: Colors.grey[900]?.withOpacity(0.8), // Slight transparency for blending with background
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: ListTile(
        leading: Icon(icon, color: Colors.white),
        title: Text(title, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
        subtitle: Text(subtitle, style: const TextStyle(color: Colors.grey)),
        trailing: const Icon(Icons.arrow_forward_ios, color: Colors.white),
        onTap: onTap,
      ),
    );
  }
}
