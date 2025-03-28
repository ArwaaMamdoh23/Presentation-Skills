// File generated by FlutterFire CLI.
// ignore_for_file: type=lint
import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

/// Default [FirebaseOptions] for use with your Firebase apps.
///
/// Example:
/// ```dart
/// import 'firebase_options.dart';
/// // ...
/// await Firebase.initializeApp(
///   options: DefaultFirebaseOptions.currentPlatform,
/// );
/// ```
class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return web;
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for macOS - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      case TargetPlatform.windows:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for Windows - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      case TargetPlatform.linux:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for Linux - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      default:
        throw UnsupportedError(
          'DefaultFirebaseOptions are not supported for this platform.',
        );
    }
  }

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: "AIzaSyAShGEW6X_IlqQ9GeOQ975PJBzIMfVhhQc",
    authDomain: "speech-analysis-db.firebaseapp.com",
    projectId: "speech-analysis-db",
    storageBucket: "speech-analysis-db.firebasestorage.app",
    messagingSenderId: "216786478864",
    appId: "1:216786478864:web:b94e39c6047fe8f7dbe795",
  );

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: "AIzaSyBaFX3feLLiajELhxOQBY4IyAzNDIqnHxU",
    appId: "1:216786478864:android:6f592ac9af7b10fbdbe795",
    messagingSenderId: "216786478864",
    projectId: "speech-analysis-db",
    storageBucket: "speech-analysis-db.firebasestorage.app",
  );

  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: "AIzaSyCfKlaPTB5KSiypmr18IbMcYjBYc--k4rw",
    appId: "1:216786478864:ios:75c16c40efc06acddbe795",
    messagingSenderId: "216786478864",
    projectId: "speech-analysis-db",
    storageBucket: "speech-analysis-db.firebasestorage.app",
    iosBundleId: "com.example.myFlutterProject",
  );
}
