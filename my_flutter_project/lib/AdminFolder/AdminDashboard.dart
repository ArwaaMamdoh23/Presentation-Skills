import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:easy_localization/easy_localization.dart'; // Import easy_localization
import '../widgets/background_wrapper.dart';

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
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          'PresentSense', 
          style: TextStyle(
            color: Colors.white, 
            fontWeight: FontWeight.bold, 
            fontSize: 28,
          ),
        ),
        centerTitle: false, 
      ),
      drawer: const AdminDrawer(),
      body: BackgroundWrapper(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 20), 

               Padding(
                padding: EdgeInsets.symmetric(horizontal: 16.0),
                child: Center(
                  child: Text(
                    'Admin Dashboard'.tr(), // Localize text
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
              ),
              const SizedBox(height: 20),
              const SystemOverview(),
              const UserManagement(),
              const PresentationAnalysisSummary(),
              const AIInsights(),
              const Footer(),
            ],
          ),
        ),
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
              color: Colors.white,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                CircleAvatar(
                  radius: 30,
                  backgroundColor: Colors.white,
                  child: Icon(Icons.person, size: 40, color: Colors.black),
                ),
                SizedBox(height: 10),
                Text(
                  'Admin Name',
                  style: TextStyle(color: Colors.black, fontSize: 18),
                ),
              ],
            ),
          ),
          ListTile(
            title: Text('User Management'.tr()), // Localize text
            onTap: () {},
          ),
          ListTile(
            title: Text('System Overview'.tr()), // Localize text
            onTap: () {},
          ),
          ListTile(
            title: Text('Reports'.tr()), // Localize text
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
      title: 'System Overview'.tr(), // Localize text
      children: [
        _buildRow(Icons.people, 'Total Active Users'.tr(), '125'),
        _buildRow(Icons.video_library, 'Total Analyses'.tr(), '250'),
        _buildRow(Icons.check_circle, 'System Status'.tr(), 'All Systems Operational', Colors.green),
      ],
    );
  }
}

class UserManagement extends StatelessWidget {
  const UserManagement({super.key});

  @override
  Widget build(BuildContext context) {
    return _buildCard(
      title: 'User Management'.tr(), // Localize text
      children: [
        _buildUserTile('Arwaa Mamdoh', 'Active'.tr()),
        _buildUserTile('Mostafa Wael', 'Inactive'.tr()),
      ],
    );
  }
}

class PresentationAnalysisSummary extends StatelessWidget {
  const PresentationAnalysisSummary({super.key});

  @override
  Widget build(BuildContext context) {
    return _buildCard(
      title: 'Recent Presentation Analysis'.tr(), // Localize text
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
      title: 'AI Insights & System Performance'.tr(), // Localize text
      children: [
        _buildRow(Icons.insights, 'AI Performance'.tr(), 'Accuracy: 92%'),
        _buildRow(Icons.trending_up, 'Trending Issues'.tr(), 'Improve tone detection'),
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
