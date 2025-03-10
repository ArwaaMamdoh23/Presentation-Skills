import 'package:flutter/material.dart';

class BackgroundWrapper extends StatelessWidget {
  final Widget child; // This will hold the page content

  const BackgroundWrapper({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // ✅ Fixed Background Image
        Container(
          decoration: BoxDecoration(
              image: DecorationImage(
                // image: const AssetImage('assets/images/Present1.jpg'),
                image: const AssetImage('assets/images/Present2.jpg'),
                fit: BoxFit.cover,
                colorFilter: ColorFilter.mode(
                  Colors.black.withOpacity(0.5),
                  BlendMode.darken,
                ),
            ),
          ),
        ),
        // ✅ Page Content (Wrapped Inside)
        child,
      ],
    );
  }
}
