import 'package:flutter/material.dart';
import '../widgets/custom_app_bar.dart'; // ✅ Import Custom AppBar
import '../widgets/background_wrapper.dart'; // ✅ Import BackgroundWrapper
import '../widgets/CustomDrawer .dart'; 

class ReportsPage extends StatelessWidget {
  ReportsPage({super.key});

  // ✅ Simulated List of Reports (Replace with actual data)
  final List<Map<String, String>> reports = [
    {"title": "Presentation 1", "date": "March 10, 2025"},
    {"title": "Presentation 2", "date": "March 8, 2025"},
    {"title": "Presentation 3", "date": "March 5, 2025"},
  ]; // ✅ If empty, it will show "No Reports Yet"

  @override
  Widget build(BuildContext context) {
    bool isUserSignedIn = true; // ✅ Change based on user authentication status

    return Scaffold(
      extendBodyBehindAppBar: true, // ✅ Extends body behind AppBar
      appBar: CustomAppBar(
        showSignIn: false, // ✅ Hide sign-in button when signed in
        isUserSignedIn: isUserSignedIn,
      ),
      drawer: CustomDrawer(isSignedIn: isUserSignedIn), // ✅ Sidebar

      body: BackgroundWrapper( // ✅ Background applied
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(height: kToolbarHeight + 20), // ✅ Push title below AppBar
                const Text(
                  'Presentation Reports',
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
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 20),
                
                reports.isEmpty // ✅ Check if reports exist
                    ? const Text(
                        'No Reports Yet',
                        style: TextStyle(color: Colors.white70, fontSize: 18),
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

  // ✅ Report Card UI
  Widget _buildReportCard({required String title, required String date, required VoidCallback onTap}) {
    return Card(
      color: Colors.white.withOpacity(0.9),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      margin: const EdgeInsets.symmetric(vertical: 10),
      child: ListTile(
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text("Date: $date"),
        leading: const Icon(Icons.insert_drive_file, color: Colors.blueAccent), // ✅ Document Icon
        trailing: const Icon(Icons.arrow_forward_ios, size: 18), // ✅ Open Icon
        onTap: onTap, // ✅ Open report when tapped
      ),
    );
  }

  // ✅ Function to Open a Report
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
              child: const Text("Close"),
            ),
          ],
        );
      },
    );
  }
}






// import 'package:flutter/material.dart';
// import '../widgets/custom_app_bar.dart'; // ✅ Import Custom AppBar
// import '../widgets/background_wrapper.dart'; // ✅ Import BackgroundWrapper
// import '../widgets/CustomDrawer .dart'; 

// class ReportsPage extends StatelessWidget {
//   const ReportsPage({super.key});

//   @override
//   Widget build(BuildContext context) {
//     bool isUserSignedIn = true; // ✅ Change based on user authentication status

//     return Scaffold(
//       extendBodyBehindAppBar: true, // ✅ Extends body behind AppBar
//       appBar: CustomAppBar(
//         showSignIn: false, // ✅ Hide sign-in button when signed in
//         isUserSignedIn: isUserSignedIn,
//       ),
//       drawer: CustomDrawer(isSignedIn: isUserSignedIn), // ✅ Sidebar on the RIGHT

//       body: BackgroundWrapper( // ✅ Add background
//         child: Center(
//           child: Padding(
//             padding: const EdgeInsets.all(16.0),
//             child: Column(
//               mainAxisAlignment: MainAxisAlignment.center,
//               children: [
//                 const Text(
//                   'Presentation Reports',
//                   style: TextStyle(
//                   color: Colors.white,
//                   fontSize: 26,
//                   fontWeight: FontWeight.bold,
//                   shadows: [
//                     Shadow(
//                       blurRadius: 3.0,
//                       color: Colors.white54,
//                       offset: Offset(0, 0),
//                     ),
//                   ],
//                 ),
//                   textAlign: TextAlign.center,
//                 ),
//                 const SizedBox(height: 20),
//                 const Text(
//                   'View insights and detailed feedback on your presentations.',
//                   style: TextStyle(color: Colors.white70, fontSize: 18),
//                   textAlign: TextAlign.center,
//                 ),
//                 const SizedBox(height: 40),
//                 ElevatedButton(
//                   onPressed: () {
//                     // ✅ Navigate to detailed reports or download reports
//                   },
//                   style: ElevatedButton.styleFrom(
//                     backgroundColor: Colors.blueAccent,
//                     padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
//                     shape: RoundedRectangleBorder(
//                       borderRadius: BorderRadius.circular(20),
//                     ),
//                   ),
//                   child: const Text(
//                     'View Reports',
//                     style: TextStyle(fontSize: 18, color: Colors.white),
//                   ),
//                 ),
//               ],
//             ),
//           ),
//         ),
//       ),
//     );
//   }
// }
