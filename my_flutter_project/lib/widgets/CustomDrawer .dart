import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class CustomDrawer extends StatelessWidget {
  final bool isSignedIn;

  const CustomDrawer({Key? key, required this.isSignedIn}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: SafeArea(
        child: Column(
          children: [
            ListTile(
              leading: const Icon(Icons.person),
              title: const Text('Profile'),
              onTap: () {
                Navigator.pushNamed(context, '/profile');
              },
            ),
            ListTile(
              leading: const Icon(Icons.upload_file),
              title: const Text('Upload'),
              onTap: () {
                Navigator.pushNamed(context, '/upload');
              },
            ),
            ListTile(
              leading: const Icon(Icons.insert_drive_file),
              title: const Text('Reports'),
              onTap: () {
                Navigator.pushNamed(context, '/report');
              },
            ),

             ListTile(
              leading: const Icon(Icons.help),
              title: const Text('Instructions'),
              onTap: () {
                Navigator.pushNamed(context, '/instructions'); 
              },
            ),

            const Spacer(),

            
            ListTile(
              leading: const Icon(Icons.settings),
              title: const Text('Settings'),
              onTap: () {
                Navigator.pushNamed(context, '/settings');
              },
            ),

            ListTile(
              leading: const Icon(Icons.info),
              title: const Text('About Us'),
              onTap: () {
                Navigator.pushNamed(context, '/aboutus'); // Change to actual settings route
              },
            ),

            if (isSignedIn)
              Align(
                alignment: Alignment.bottomCenter,
                child: ListTile(
                  leading: const Icon(Icons.exit_to_app, color: Colors.red),
                  title: const Text(
                    'Log Out',
                    style: TextStyle(color: Colors.red),
                  ),
                 onTap: () async {
  try {
    await Supabase.instance.client.auth.signOut();
    Navigator.pushNamedAndRemoveUntil(context, '/home', (route) => false);
  } catch (e) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Logout failed: $e')),
    );
  }
},
                ),
              ),
          ],
        ),
      ),
    );
  }
}
