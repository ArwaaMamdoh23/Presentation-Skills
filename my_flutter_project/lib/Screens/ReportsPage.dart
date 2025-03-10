// import 'package:flutter/material.dart';
// import '../widgets/custom_app_bar.dart'; // ✅ Import the Custom AppBar

// class ReportsPage extends StatelessWidget {
//   const ReportsPage({super.key});

//   @override
//   Widget build(BuildContext context) {
//     bool isUserSignedIn = true; // ✅ Change based on user auth status

//     return Scaffold(
//       extendBodyBehindAppBar: true,
//       appBar: CustomAppBar(), // ✅ Use custom app bar
//       body: Center(
//         child: Padding(
//           padding: const EdgeInsets.all(16.0),
//           child: Column(
//             mainAxisAlignment: MainAxisAlignment.center,
//             children: [
//               const Text(
//                 'Presentation Reports',
//                 style: TextStyle(
//                   color: Colors.white,
//                   fontSize: 28,
//                   fontWeight: FontWeight.bold,
//                 ),
//                 textAlign: TextAlign.center,
//               ),
//               const SizedBox(height: 20),
//               const Text(
//                 'View insights and detailed feedback on your presentations.',
//                 style: TextStyle(color: Colors.white70, fontSize: 18),
//                 textAlign: TextAlign.center,
//               ),
//               const SizedBox(height: 40),
//               ElevatedButton(
//                 onPressed: () {
//                   // ✅ Navigate to detailed reports or download reports
//                 },
//                 style: ElevatedButton.styleFrom(
//                   backgroundColor: Colors.blueAccent,
//                   padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
//                   shape: RoundedRectangleBorder(
//                     borderRadius: BorderRadius.circular(20),
//                   ),
//                 ),
//                 child: const Text(
//                   'View Reports',
//                   style: TextStyle(fontSize: 18, color: Colors.white),
//                 ),
//               ),
//             ],
//           ),
//         ),
//       ),
//     );
//   }
// }
