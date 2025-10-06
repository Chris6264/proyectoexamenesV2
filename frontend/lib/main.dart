import 'package:flutter/material.dart';
import 'screens/omr_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Evalink',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const OmrScreen(),
    );
  }
}
