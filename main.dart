import 'package:flutter/material.dart';

import 'package:project_mobile/Screens/journal.dart';
import 'Screens/welcomescreen (1).dart';

import 'Screens/contactus.dart';
import 'Screens/journal.dart';
import 'Home.dart';
import 'Screens/profile.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';

//void main() => runApp(MyApp());
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        // This makes the visual density adapt to the platform that yosu run
        // the app on. For desktop platforms, the controls will be smaller and
        // closer together (more dense) than on mobile platforms.
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: WelcomeScreen(),
      routes: {
        'home': (context) => HomePage(),
        'Contact Us': (context) => ContactUs(),
        'profile': (context) => ProfilePage(),
        'addjournal': (context) => Journal(),
      },
    );
  }
}
