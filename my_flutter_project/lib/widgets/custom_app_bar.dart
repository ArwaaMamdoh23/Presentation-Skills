import 'package:flutter/material.dart';

class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  final bool showSignIn;
  final bool isUserSignedIn;
  final bool hideSignInButton;  // New parameter to hide button on sign-in page

  const CustomAppBar({
    super.key,
    required this.showSignIn,
    required this.isUserSignedIn,
    this.hideSignInButton = false,  // Default value to show sign-in button
  });

  @override
  Widget build(BuildContext context) {
    return AppBar(
      backgroundColor: Colors.transparent,
      elevation: 0,
      title: GestureDetector(
        onTap: () {
          Navigator.pushNamed(context, '/home'); // Navigate to Home Page
        },
        child: const Text(
          "PresentSense",
          style: TextStyle(
            color: Colors.white,
            fontSize: 28,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      centerTitle: false, // Aligns title to the left
      actions: [
        // 🔹 **Fix: Separated `hideSignInButton` logic from `showSignIn` check**
        if (showSignIn && !hideSignInButton) 
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: ElevatedButton(
              onPressed: () {
                Navigator.pushNamed(context, '/sign-in');
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.transparent,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(25),
                ),
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              ),
              child: const Text(
                "Sign In",
                style: TextStyle(fontSize: 18, color: Colors.white),
              ),
            ),
          ),

        if (isUserSignedIn) 
          IconButton(
            icon: const Icon(Icons.person, color: Colors.white), 
            iconSize: 40, 
            onPressed: () {
              Navigator.pushNamed(context, '/profile'); // Navigate to Profile Page
            },
          ),
      ],
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}
