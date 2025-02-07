import 'package:flutter/material.dart';

class SettingsPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Settings"),
        backgroundColor: Colors.black,
      ),
      body: Container(
        color: Colors.black, // Dark theme background
        padding: EdgeInsets.all(16),
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
    );
  }

  Widget _buildSettingsItem({required IconData icon, required String title, required String subtitle, required VoidCallback onTap}) {
    return Card(
      color: Colors.grey[900],
      child: ListTile(
        leading: Icon(icon, color: Colors.white),
        title: Text(title, style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
        subtitle: Text(subtitle, style: TextStyle(color: Colors.grey)),
        trailing: Icon(Icons.arrow_forward_ios, color: Colors.white),
        onTap: onTap,
      ),
    );
  }
}
