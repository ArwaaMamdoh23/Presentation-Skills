import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ProfilePage extends StatefulWidget {
  @override
  _ProfilePageState createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  String? name;
  String? email;
  String? profession;

  @override
  void initState() {
    super.initState();
    _loadUserProfile();
  }

  // Load user profile data from SharedPreferences
  void _loadUserProfile() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      name = prefs.getString('name') ?? 'User';
      email = prefs.getString('email') ?? 'user@example.com';
      profession = prefs.getString('profession') ?? 'Student';
    });
  }

  // Save user profile data to SharedPreferences
  void _saveUserProfile(String name, String email, String profession) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setString('name', name);
    prefs.setString('email', email);
    prefs.setString('profession', profession);
  }

  // Edit profile
  void _editProfile() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        TextEditingController nameController = TextEditingController(text: name);
        TextEditingController emailController = TextEditingController(text: email);
        TextEditingController professionController = TextEditingController(text: profession);

        return AlertDialog(
          title: Text('Edit Profile'),
          content: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              TextField(
                controller: nameController,
                decoration: InputDecoration(labelText: 'Name'),
              ),
              TextField(
                controller: emailController,
                decoration: InputDecoration(labelText: 'Email'),
              ),
              TextField(
                controller: professionController,
                decoration: InputDecoration(labelText: 'Profession'),
              ),
            ],
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () {
                _saveUserProfile(nameController.text, emailController.text, professionController.text);
                Navigator.of(context).pop();
                _loadUserProfile();
              },
              child: Text('Save'),
            ),
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: Text('Cancel'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('User Profile'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            // Profile Info
            CircleAvatar(
              radius: 50,
              backgroundColor: Colors.blueAccent,
              child: Text(
                name != null ? name![0] : 'U', // Initials
                style: TextStyle(fontSize: 40, color: Colors.white),
              ),
            ),
            SizedBox(height: 20),
            Text('Name: $name', style: TextStyle(fontSize: 18)),
            Text('Email: $email', style: TextStyle(fontSize: 18)),
            Text('Profession: $profession', style: TextStyle(fontSize: 18)),

            SizedBox(height: 30),
            // Edit Profile Button
            ElevatedButton(
              onPressed: _editProfile,
              child: Text('Edit Profile'),
            ),

            SizedBox(height: 30),
            // Progress Tracking Section
            Text(
              'Progress Tracking',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 10),
            // Example Progress Bar (You can add more data like performance score, feedback, etc.)
            LinearProgressIndicator(value: 0.7), // 70% progress as an example
            SizedBox(height: 10),
            Text('Current Progress: 70%'),

            SizedBox(height: 30),
            // Achievements Section
            Text(
              'Achievements',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 10),
            // Example Achievements List
            ListTile(
              title: Text('Completed 10 Presentations'),
              subtitle: Text('Awarded: Bronze Level'),
            ),
            ListTile(
              title: Text('Improved Speech Clarity by 15%'),
              subtitle: Text('Awarded: Silver Level'),
            ),
          ],
        ),
      ),
    );
  }
}
