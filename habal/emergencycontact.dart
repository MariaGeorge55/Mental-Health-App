import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';

import '../widgets/emergencycontact/emergency_contacts.dart';
import '../widgets/emergencycontact/addemergencycontactbutton.dart';

class ViewEmergencyContact extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xffe0ecde),
      appBar: AppBar(
        backgroundColor: const Color(0xff68b2a0),
//            title: Center(child: Image(image: AssetImage('images/Icon.png'))),
      ),
      body: Column(
        children: [
          Addemergencybutton(),
          Listemergencycontacts(),
        ],
      ),
    );
  }
}
