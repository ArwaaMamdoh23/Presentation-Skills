import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import '../widgets/custom_app_bar.dart';
import '../widgets/background_wrapper.dart';
import '../widgets/CustomDrawer.dart';

class ReportsPage extends StatelessWidget {
  ReportsPage({super.key});

  final List<Map<String, String>> reports = [
    {"title": "Presentation 1", "date": "March 10, 2025"},
    {"title": "Presentation 2", "date": "March 8, 2025"},
    {"title": "Presentation 3", "date": "March 5, 2025"},
  ];

  @override
  Widget build(BuildContext context) {
    bool isUserSignedIn = true;
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: CustomAppBar(
        showSignIn: false,
        isUserSignedIn: isUserSignedIn,
      ),
      drawer: CustomDrawer(isSignedIn: isUserSignedIn),
      body: BackgroundWrapper(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(height: kToolbarHeight + 20),
                Text(
                  'presentation_reports'.tr(), // Translated text key for 'Presentation Reports'
                  style: const TextStyle(
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
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),

                reports.isEmpty
                    ? Text(
                        'no_reports_yet'.tr(), // Translated text key for 'No Reports Yet'
                        style: const TextStyle(color: Colors.white70, fontSize: 18),
                      )
                    : Expanded(
                        child: ListView.builder(
                          itemCount: reports.length,
                          itemBuilder: (context, index) {
                            final report = reports[index];
                            return _buildReportCard(
                              title: report["title"]!,
                              date: report["date"]!,
                              onTap: () {
                                _openReport(context, report["title"]!);
                              },
                            );
                          },
                        ),
                      ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildReportCard({required String title, required String date, required VoidCallback onTap}) {
    return Card(
      color: Colors.white.withOpacity(0.9),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      margin: const EdgeInsets.symmetric(vertical: 10),
      child: ListTile(
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text("date: $date".tr()), // Translated text key for 'Date:'
        leading: const Icon(Icons.insert_drive_file, color: Colors.blueAccent),
        trailing: const Icon(Icons.arrow_forward_ios, size: 18),
        onTap: onTap,
      ),
    );
  }

  void _openReport(BuildContext context, String reportTitle) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text(reportTitle),
          content: const Text("This is where the detailed report will be displayed."),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text ('close'.tr()), // Translated text key for 'Close'
            ),
          ],
        );
      },
    );
  }
}
