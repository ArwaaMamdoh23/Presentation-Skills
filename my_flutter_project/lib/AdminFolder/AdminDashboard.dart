import 'package:flutter/material.dart';
import 'dart:ui';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: AdminDashboard(),
    );
  }
}

class AdminDashboard extends StatelessWidget {
  const AdminDashboard({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: Stack(
        children: [
          // Background with blur effect
          Container(
            decoration: BoxDecoration(
              image: DecorationImage(
                image: const AssetImage('assets/images/back.jpg'),
                fit: BoxFit.cover,
                colorFilter: ColorFilter.mode(
                  Colors.black.withOpacity(0.4),
                  BlendMode.darken,
                ),
              ),
            ),
            child: ClipRect(
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 5, sigmaY: 5),
                child: Container(color: Colors.transparent),
              ),
            ),
          ),

          Scaffold(
            backgroundColor: Colors.transparent,
            appBar: AppBar(
              title: const Text('Admin Dashboard'),
              backgroundColor: Colors.transparent,
              elevation: 0,
              actions: [
                IconButton(
                  icon: const Icon(Icons.notifications),
                  onPressed: () {},
                ),
                IconButton(
                  icon: const Icon(Icons.settings),
                  onPressed: () {},
                ),
              ],
            ),
            drawer: const AdminDrawer(),
            body: const SingleChildScrollView(
              child: Column(
                children: [
                  SystemOverview(),
                  UserManagement(),
                  PresentationAnalysisSummary(),
                  AIInsights(),
                  Footer(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class AdminDrawer extends StatelessWidget {
  const AdminDrawer({super.key});

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          const DrawerHeader(
            decoration: BoxDecoration(
              color: Colors.blue,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                CircleAvatar(
                  radius: 30,
                  backgroundColor: Colors.blue,
                  child: Icon(Icons.person, size: 40, color: Colors.white),
                ),
                SizedBox(height: 10),
                Text(
                  'Admin Name',
                  style: TextStyle(color: Colors.white, fontSize: 18),
                ),
              ],
            ),
          ),
          ListTile(
            title: const Text('User Management'),
            onTap: () {},
          ),
          ListTile(
            title: const Text('System Overview'),
            onTap: () {},
          ),
          ListTile(
            title: const Text('Reports'),
            onTap: () {},
          ),
        ],
      ),
    );
  }
}

class SystemOverview extends StatelessWidget {
  const SystemOverview({super.key});

  @override
  Widget build(BuildContext context) {
    return _buildCard(
      title: 'System Overview',
      children: [
        _buildRow(Icons.people, 'Total Active Users', '125'),
        _buildRow(Icons.video_library, 'Total Analyses', '250'),
        _buildRow(Icons.check_circle, 'System Status', 'All Systems Operational', Colors.green),
      ],
    );
  }
}

class UserManagement extends StatelessWidget {
  const UserManagement({super.key});

  @override
  Widget build(BuildContext context) {
    return _buildCard(
      title: 'User Management',
      children: [
        _buildUserTile('Arwaa Mamdoh', 'Active'),
        _buildUserTile('Mostafa Wael', 'Inactive'),
      ],
    );
  }
}

class PresentationAnalysisSummary extends StatelessWidget {
  const PresentationAnalysisSummary({super.key});

  @override
  Widget build(BuildContext context) {
    return _buildCard(
      title: 'Recent Presentation Analysis',
      children: [
        _buildListTile(Icons.video_library, 'Presentation 1', 'Score: 8/10'),
        _buildListTile(Icons.video_library, 'Presentation 2', 'Score: 7/10'),
      ],
    );
  }
}

class AIInsights extends StatelessWidget {
  const AIInsights({super.key});

  @override
  Widget build(BuildContext context) {
    return _buildCard(
      title: 'AI Insights & System Performance',
      children: [
        _buildRow(Icons.insights, 'AI Performance', 'Accuracy: 92%'),
        _buildRow(Icons.trending_up, 'Trending Issues', 'Improve tone detection'),
      ],
    );
  }
}

class Footer extends StatelessWidget {
  const Footer({super.key});

  @override
  Widget build(BuildContext context) {  
    return const Padding(
      padding: EdgeInsets.all(16),
      child: Text('App Version 1.0', style: TextStyle(fontSize: 14, color: Colors.grey)),
    );
  }
}

// Reusable Components
Widget _buildCard({required String title, required List<Widget> children}) {
  return Card(
    margin: const EdgeInsets.all(16),
    color: Colors.white.withOpacity(0.9),
    child: Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
          const SizedBox(height: 10),
          ...children,
        ],
      ),
    ),
  );
}

Widget _buildRow(IconData icon, String title, String value, [Color? color]) {
  return Padding(
    padding: const EdgeInsets.symmetric(vertical: 5),
    child: Row(
      children: [
        Icon(icon, size: 40, color: color ?? Colors.black),
        const SizedBox(width: 10),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: const TextStyle(fontSize: 16)),
            Text(value, style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color ?? Colors.black)),
          ],
        ),
      ],
    ),
  );
}

Widget _buildUserTile(String name, String status) {
  return ListTile(
    title: Text(name),
    subtitle: Text(status),
    leading: const CircleAvatar(child: Icon(Icons.person)),
    onTap: () {},
  );
}

Widget _buildListTile(IconData icon, String title, String subtitle) {
  return ListTile(
    leading: Icon(icon),
    title: Text(title),
    subtitle: Text(subtitle),
  );
}