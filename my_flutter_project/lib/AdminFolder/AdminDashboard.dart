import 'package:flutter/material.dart';
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

  void showSupportDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Submit a Support Request'),
        content: const TextField(
          maxLines: 5,
          decoration: InputDecoration(hintText: 'Describe your issue...'),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
          ElevatedButton(
            onPressed: () {
              // TODO: implement support request sending
              Navigator.pop(context);
            },
            child: const Text('Submit'),
          ),
        ],
      ),
    );
  }

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
              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 16.0),
                child: Center(
                  child: Text(
                    'Admin Dashboard',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 26,
                      fontWeight: FontWeight.bold,
                      shadows: [Shadow(blurRadius: 3.0, color: Colors.white54)],
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              const SystemOverview(),
              const UserManagement(),
              const PresentationAnalysisSummary(),
              const AIInsights(),
              const FeedbackReviewSection(),
              SupportSection(),
              const Footer(),
            ],
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => showSupportDialog(context),
        tooltip: 'Live Support',
        child: const Icon(Icons.chat),
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
            title: Text('User Management'.tr()),
            onTap: () {},
          ),
          ListTile(
            title: Text('System Overview'.tr()),
            onTap: () {},
          ),
          ListTile(
            title: Text('Reports'.tr()),
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
      title: 'System Overview'.tr(),
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
      title: 'User Management'.tr(),
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
      title: 'Recent Presentation Analysis'.tr(),
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
      title: 'AI Insights & System Performance'.tr(),
      children: [
        _buildRow(Icons.insights, 'AI Performance'.tr(), 'Accuracy: 92%'),
        _buildRow(Icons.trending_up, 'Trending Issues'.tr(), 'Improve tone detection'),
      ],
    );
  }
}

class FeedbackReviewSection extends StatelessWidget {
  const FeedbackReviewSection({super.key});

  @override
  Widget build(BuildContext context) {
    return _buildCard(
      title: 'User Feedback Review'.tr(),
      children: [
        _buildListTile(Icons.feedback, 'User: Arwaa Mamdoh', 'Grammar: 9/10, Fluency: 8/10, Pronunciation: 8/10, Body: 7/10, Facial expressions: 8/10, Lang: English'),
        _buildListTile(Icons.feedback, 'User: Mostafa Wael', 'Grammar: 6/10, Fluency: 7/10, Pronunciation: 6/10, Body: 6/10, Facial expressions: 7/10, Lang: Arabic'),
      ],
    );
  }
}

class SupportSection extends StatelessWidget {
  const SupportSection({super.key});

  void showSupportDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Submit a Support Request'),
        content: const TextField(
          maxLines: 5,
          decoration: InputDecoration(hintText: 'Describe your issue...'),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
          ElevatedButton(
            onPressed: () {
              // TODO: implement sending logic
              Navigator.pop(context);
            },
            child: const Text('Submit'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return _buildCard(
      title: 'User Support',
      children: [
        ExpansionTile(
          title: const Text('How do I upload a presentation?'),
          children: const [
            Padding(
              padding: EdgeInsets.all(8.0),
              child: Text('Go to the Upload tab, select your video, and click Submit.'),
            ),
          ],
        ),
        ExpansionTile(
          title: const Text('Can I analyze videos in other languages?'),
          children: const [
            Padding(
              padding: EdgeInsets.all(8.0),
              child: Text('Yes, we support multilingual feedback using Whisper and Argos Translate.'),
            ),
          ],
        ),
        ExpansionTile(
          title: const Text('What does the feedback score mean?'),
          children: const [
            Padding(
              padding: EdgeInsets.all(8.0),
              child: Text('It reflects grammar, fluency, pronunciation, emotion, and body language accuracy.'),
            ),
          ],
        ),
        const ListTile(
          leading: Icon(Icons.email),
          title: Text('Contact Us'),
          subtitle: Text('support@presentsense.ai'),
        ),
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
