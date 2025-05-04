import 'package:flutter/material.dart';

class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  final bool showSignIn;
  final bool isUserSignedIn;
  final bool hideSignInButton;
  final List<Widget>? extraActions; // ✅ Add this parameter

  const CustomAppBar({
    super.key,
    required this.showSignIn,
    required this.isUserSignedIn,
    this.hideSignInButton = false,
    this.extraActions, // ✅ Include it in the constructor
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
      centerTitle: false,
      actions: [
        if (isUserSignedIn)
          IconButton(
            icon: const Icon(Icons.person, color: Colors.white),
            iconSize: 40,
            onPressed: () {
              Navigator.pushNamed(context, '/profile');
            },
          ),
        // ✅ Add any extra widgets passed from the parent
        if (extraActions != null) ...extraActions!,
      ],
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}
