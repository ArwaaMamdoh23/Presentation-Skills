import 'package:flutter/material.dart';

class CustomDrawer extends StatelessWidget {
  final bool isSignedIn;

  const CustomDrawer({Key? key, required this.isSignedIn}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: SafeArea(  // Use SafeArea here to ensure the drawer content is inside safe bounds
        child: Column(
          children: [
            // ListTile(
            //   leading: const Icon(Icons.home),
            //   title: const Text('Home'),
            //   onTap: () {
            //     Navigator.pushNamed(context, '/home');
            //   },
            // ),
            
            // Main Menu Options
            ListTile(
              leading: const Icon(Icons.person), // Profile Icon
              title: const Text('Profile'),
              onTap: () {
                Navigator.pushNamed(context, '/profile');  // Change to your actual profile page
              },
            ),
  
            ListTile(
              leading: const Icon(Icons.upload_file), // Upload Icon
              title: const Text('Upload'),
              onTap: () {
                Navigator.pushNamed(context, '/upload');  // Change to your actual page route
              },
            ),
            
            ListTile(
              leading: const Icon(Icons.insert_drive_file),
              title: const Text('Reports'),
              onTap: () {
                Navigator.pushNamed(context, '/report');  // Change to your actual page route
              },
            ),
  
            const Spacer(),  // Pushes Settings and Log Out to the bottom
  
            // Settings (Above Log Out)
            ListTile(
              leading: const Icon(Icons.settings),
              title: const Text('Settings'),
              onTap: () {
                Navigator.pushNamed(context, '/settings'); // Change to actual settings route
              },
            ),
  
            // Log Out (At the Bottom)
            if (isSignedIn)
              Align(
                alignment: Alignment.bottomCenter,
                child: ListTile(
                  leading: const Icon(Icons.exit_to_app, color: Colors.red),
                  title: const Text('Log Out', style: TextStyle(color: Colors.red)),
                  onTap: () {
                    // Handle log out logic
                    Navigator.pushNamed(context, '/home');
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// import 'package:flutter/material.dart';

// class CustomDrawer extends StatelessWidget {
//   final bool isSignedIn;

//   const CustomDrawer({Key? key, required this.isSignedIn}) : super(key: key);

//   @override
//   Widget build(BuildContext context) {
//     return Drawer(
//       child: Column(
//         children: [
//           ListTile(
//             leading: const Icon(Icons.home),
//             title: const Text('Home'),
//             onTap: () {
//               Navigator.pushNamed(context, '/home');
//             },
//           ),
          
//           // ✅ Main Menu Options
//           ListTile(
//             leading: const Icon(Icons.person), // 👤 Profile Icon
//             title: const Text('Profile'),
//             onTap: () {
//               Navigator.pushNamed(context, '/profile');  // Change to your actual profile page
//             },
//           ),

          

//           ListTile(
//             leading: const Icon(Icons.upload_file), // 📤 Upload Icon
//             title: const Text('Upload'),
//             onTap: () {
//               Navigator.pushNamed(context, '/upload');  // Change to your actual page route
//             },
//           ),
          
//           ListTile(
//             leading: const Icon(Icons.insert_drive_file),
//             title: const Text('Reports'),
//             onTap: () {
//               Navigator.pushNamed(context, '/report');  // Change to your actual page route
//             },
//           ),

//           const Spacer(),  // Pushes Settings and Log Out to the bottom

//           // ✅ Settings (Above Log Out)
//           ListTile(
//             leading: const Icon(Icons.settings),
//             title: const Text('Settings'),
//             onTap: () {
//               Navigator.pushNamed(context, '/settings'); // Change to actual settings route
//             },
//           ),

//           // ✅ Log Out (At the Bottom)
//           if (isSignedIn)
//             Align(
//               alignment: Alignment.bottomCenter,
//               child: ListTile(
//                 leading: const Icon(Icons.exit_to_app, color: Colors.red),
//                 title: const Text('Log Out', style: TextStyle(color: Colors.red)),
//                 onTap: () {
//                   // Handle log out logic
//                   Navigator.pushNamed(context, '/home');
//                 },
//               ),
//             ),
//         ],
//       ),
//     );
//   }
// }
