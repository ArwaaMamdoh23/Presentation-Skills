import 'package:flutter/material.dart';
import 'package:my_flutter_project/Screens/HomePage.dart';
import '../widgets/custom_app_bar.dart'; // ✅ Import Custom AppBar
import '../widgets/background_wrapper.dart'; // ✅ Import Background Wrapper
// import 'package:my_flutter_project/Screens/SignInPage.dart'; // ✅ Import Sign-In Page for navigation
import '../widgets/CustomDrawer .dart'; 

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    bool isUserSignedIn = true; // ✅ Change based on user authentication status

    return Scaffold(
      extendBodyBehindAppBar: true, // ✅ Extends content behind AppBar
      appBar: CustomAppBar(
        showSignIn: false,
        isUserSignedIn: true, // ✅ User is signed in, show Profile & Settings
      ),
      drawer: CustomDrawer(isSignedIn: isUserSignedIn), // ✅ Sidebar on the RIGHT

      body: BackgroundWrapper( // ✅ Apply fixed background
        child: Column(
          children: [
            const SizedBox(height: kToolbarHeight), // ✅ Ensure content doesn't overlap AppBar
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 20),
              child: Text(
                "Settings",
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
                        _showDeleteConfirmationDialog(context);
                        // Implement account deletion confirmation
                      },
                    ),
                    _buildSettingsItem(
                      icon: Icons.logout,
                      title: "Sign Out",
                      subtitle: "Log out from your account",
                      onTap: () {
                        _signOut(context);
                      },
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSettingsItem({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Card(
      color: Colors.grey[900]?.withOpacity(0.8), // ✅ Slight transparency
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: ListTile(
        leading: Icon(icon, color: Colors.white),
        title: Text(
          title,
          style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
        ),
        subtitle: Text(subtitle, style: const TextStyle(color: Colors.grey)),
        trailing: const Icon(Icons.arrow_forward_ios, color: Colors.white),
        onTap: onTap,
      ),
    );
  }

  void _signOut(BuildContext context) {
    // Perform sign-out logic (e.g., FirebaseAuth.instance.signOut();)
    
    // Navigate to the sign-in page
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => const HomePage()),
    );
  }

  void _showDeleteConfirmationDialog(BuildContext context) {
  showDialog(
    context: context,
    builder: (BuildContext context) {
      return AlertDialog(
        title: const Text("Confirm Deletion"),
        content: const Text("Are you sure you want to permanently delete your account? This action cannot be undone."),
        actions: [
          // Cancel Button
          TextButton(
            onPressed: () {
              Navigator.of(context).pop(); // Close the dialog
            },
            child: const Text("Cancel", style: TextStyle(color: Colors.blue)),
          ),
          
          // Confirm Delete Button
          TextButton(
            onPressed: () {
              Navigator.of(context).pop(); // Close the dialog
              // _deleteAccount(); // Call the delete function
            },
            child: const Text("Delete", style: TextStyle(color: Colors.red)),
          ),
        ],
        
      );
        // Navigator.pushNamed(context, '/home');

    },
  );
}

// void _deleteAccount() {
//   // TODO: Implement account deletion logic (Firebase Auth, API request, etc.)

//   // After deletion, navigate to Home
//   Navigator.pushNamedAndRemoveUntil(context, '/home', (route) => false);
// }


}
