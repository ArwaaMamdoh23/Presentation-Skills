import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: AdminDashboard(),
    );
  }
}

class AdminDashboard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Admin Dashboard'),
        actions: [
          IconButton(
            icon: Icon(Icons.notifications),
            onPressed: () {
              // Handle notifications
            },
          ),
          IconButton(
            icon: Icon(Icons.settings),
            onPressed: () {
              // Handle settings
            },
          ),
        ],
      ),
      drawer: AdminDrawer(), // Side navigation menu
      body: SingleChildScrollView(
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
    );
  }
}

class AdminDrawer extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: ListView(
        padding: EdgeInsets.zero,
        children: [
          DrawerHeader(
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
            decoration: BoxDecoration(
              color: Colors.blue,
            ),
          ),
          ListTile(
            title: Text('User Management'),
            onTap: () {
              // Navigate to user management screen
            },
          ),
          ListTile(
            title: Text('System Overview'),
            onTap: () {
              // Navigate to system overview screen
            },
          ),
          ListTile(
            title: Text('Reports'),
            onTap: () {
              // Navigate to reports screen
            },
          ),
        ],
      ),
    );
  }
}

class SystemOverview extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.all(16),
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('System Overview', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            SizedBox(height: 10),
            Row(
              children: [
                Icon(Icons.people, size: 40),
                SizedBox(width: 10),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Total Active Users', style: TextStyle(fontSize: 16)),
                    Text('125', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  ],
                ),
              ],
            ),
            SizedBox(height: 10),
            Row(
              children: [
                Icon(Icons.video_library, size: 40),
                SizedBox(width: 10),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Total Analyses', style: TextStyle(fontSize: 16)),
                    Text('250', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  ],
                ),
              ],
            ),
            SizedBox(height: 10),
            Row(
              children: [
                Icon(Icons.check_circle, size: 40),
                SizedBox(width: 10),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('System Status', style: TextStyle(fontSize: 16)),
                    Text('All Systems Operational', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.green)),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class UserManagement extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.all(16),
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('User Management', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            SizedBox(height: 10),
            ListTile(
              title: Text('John Doe'),
              subtitle: Text('Active'),
              leading: CircleAvatar(child: Icon(Icons.person)),
              onTap: () {
                // View detailed user info
              },
            ),
            ListTile(
              title: Text('Jane Smith'),
              subtitle: Text('Inactive'),
              leading: CircleAvatar(child: Icon(Icons.person)),
              onTap: () {
                // View detailed user info
              },
            ),
            // Add more user list items here
          ],
        ),
      ),
    );
  }
}

class PresentationAnalysisSummary extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.all(16),
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Recent Presentation Analysis', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            SizedBox(height: 10),
            ListTile(
              leading: Icon(Icons.video_library),
              title: Text('Presentation 1'),
              subtitle: Text('Score: 8/10'),
            ),
            ListTile(
              leading: Icon(Icons.video_library),
              title: Text('Presentation 2'),
              subtitle: Text('Score: 7/10'),
            ),
            // Add more presentation analysis items here
          ],
        ),
      ),
    );
  }
}

class AIInsights extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.all(16),
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('AI Insights & System Performance', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            SizedBox(height: 10),
            Row(
              children: [
                Icon(Icons.insights, size: 40),
                SizedBox(width: 10),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('AI Performance', style: TextStyle(fontSize: 16)),
                    Text('Accuracy: 92%', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  ],
                ),
              ],
            ),
            SizedBox(height: 10),
            Row(
              children: [
                Icon(Icons.trending_up, size: 40),
                SizedBox(width: 10),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Trending Issues', style: TextStyle(fontSize: 16)),
                    Text('Improve tone detection', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class Footer extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.all(16),
      child: Text('App Version 1.0', style: TextStyle(fontSize: 14, color: Colors.grey)),
    );
  }
}
