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
