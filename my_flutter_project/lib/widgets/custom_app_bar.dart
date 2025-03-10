import 'package:flutter/material.dart';

class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  final bool showSignIn;
  final bool isUserSignedIn;

  const CustomAppBar({
    super.key,
    required this.showSignIn,
    required this.isUserSignedIn,
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
        if (showSignIn && !isUserSignedIn)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: ElevatedButton(
              onPressed: () {
                Navigator.pushNamed(context, '/sign-in');
              },
              style: ElevatedButton.styleFrom(
                backgroundColor:Colors.transparent,
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

        if (isUserSignedIn) ...[
          // ✅ Settings Icon
          IconButton(
            icon: const Icon(Icons.settings, color: Colors.white), // ✅ Standard Settings Icon
            onPressed: () {
              Navigator.pushNamed(context, '/settings'); // Navigate to Settings Page
            },
          ),
          // ✅ Profile Icon
          IconButton(
            icon: const Icon(Icons.person, color: Colors.white), // ✅ Standard Profile Icon
            onPressed: () {
              Navigator.pushNamed(context, '/profile'); // Navigate to Profile Page
            },
          ),
        ],
      ],
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}
