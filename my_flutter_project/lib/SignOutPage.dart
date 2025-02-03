// // import 'package:flutter/material.dart';

// // class SignOutPage extends StatelessWidget {
// //   @override
// //   Widget build(BuildContext context) {
// //     return Scaffold(
// //       backgroundColor: Colors.transparent,
// //       body: Stack(
// //         children: [
// //           // Background Gradient
// //           Container(
// //             decoration: BoxDecoration(
// //               gradient: LinearGradient(
// //                 colors: [Colors.blue.shade800, Colors.blue.shade300],
// //                 begin: Alignment.topLeft,
// //                 end: Alignment.bottomRight,
// //               ),
// //             ),
// //           ),
// //           // Centered content
// //           Center(
// //             child: Padding(
// //               padding: const EdgeInsets.symmetric(horizontal: 30.0),
// //               child: Column(
// //                 mainAxisAlignment: MainAxisAlignment.center,
// //                 children: <Widget>[
// //                   Text(
// //                     'You have been signed out!',
// //                     style: TextStyle(
// //                       color: Colors.white,
// //                       fontSize: 30,
// //                       fontWeight: FontWeight.bold,
// //                     ),
// //                   ),
// //                   SizedBox(height: 40),
// //                   ElevatedButton(
// //                     onPressed: () {
// //                       // Redirect to AuthPage (sign-in page) or your desired page
// //                       Navigator.pushReplacementNamed(context, '/');
// //                     },
// //                     style: ElevatedButton.styleFrom(
// //                       backgroundColor: Colors.red.shade900,
// //                       shape: RoundedRectangleBorder(
// //                         borderRadius: BorderRadius.circular(30),
// //                       ),
// //                       padding: EdgeInsets.symmetric(vertical: 15),
// //                     ),
// //                     child: Text(
// //                       'Sign In Again',
// //                       style: TextStyle(fontSize: 18),
// //                     ),
// //                   ),
// //                 ],
// //               ),
// //             ),
// //           ),
// //         ],
// //       ),
// //     );
// //   }
// // }

// import 'package:flutter/material.dart';

// class SignOutPage extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       backgroundColor: Colors.transparent,
//       body: Stack(
//         children: [
//           // Background gradient
//           Container(
//             decoration: BoxDecoration(
//               gradient: LinearGradient(
//                 colors: [Colors.blue.shade800, Colors.blue.shade300],
//                 begin: Alignment.topLeft,
//                 end: Alignment.bottomRight,
//               ),
//             ),
//           ),
//           // Centered content
//           Center(
//             child: Padding(
//               padding: const EdgeInsets.symmetric(horizontal: 30.0),
//               child: Column(
//                 mainAxisAlignment: MainAxisAlignment.center,
//                 children: <Widget>[
//                   Text(
//                     'You have been signed out!',
//                     style: TextStyle(
//                       color: Colors.white,
//                       fontSize: 30,
//                       fontWeight: FontWeight.bold,
//                     ),
//                   ),
//                   SizedBox(height: 40),
//                   ElevatedButton(
//                     onPressed: () {
//                       // Navigate back to AuthPage (sign-in page)
//                       Navigator.pushReplacementNamed(context, '/');
//                     },
//                     style: ElevatedButton.styleFrom(
//                       backgroundColor: Colors.red.shade900,
//                       shape: RoundedRectangleBorder(
//                         borderRadius: BorderRadius.circular(30),
//                       ),
//                       padding: EdgeInsets.symmetric(vertical: 15),
//                     ),
//                     child: Text(
//                       'Sign In Again',
//                       style: TextStyle(fontSize: 18),
//                     ),
//                   ),
//                 ],
//               ),
//             ),
//           ),
//         ],
//       ),
//     );
//   }
// }
