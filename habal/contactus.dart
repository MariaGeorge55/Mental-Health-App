import 'package:flutter/material.dart';
import '../widgets/Contactus/contactusform.dart';
import '../widgets/Contactus/titletext.dart';

class ContactUs extends StatefulWidget {
  @override
  _ContactUsState createState() => _ContactUsState();
}

class _ContactUsState extends State<ContactUs> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Contact Us '),
        backgroundColor: const Color(0xff68b2a0),
      ),
      backgroundColor: const Color(0xffe0ecde),
      body: SafeArea(
        child: Container(
            child: new SingleChildScrollView(
          child: Column(children: [
            TitleText(),
            ContactusForm(),
          ]),
        )),
      ),
    );
  }
}
