import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import '../widgets/Addemergencycontact/AddemergencycontactForm.dart';

class Addemergencycontact extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    var row1 = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Padding(padding: EdgeInsets.only(top: 50.0)),
      ],
    );
    var row2 = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          'Enter your emergency contact details',
          style: TextStyle(
              fontSize: 15, color: Colors.green, fontWeight: FontWeight.bold),
        ),
      ],
    );
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xff68b2a0),
        title: Center(child: Image(image: AssetImage('images/Icon.png'))),
      ),
      body: Center(
          child: Container(
              child: new SingleChildScrollView(
        child: Column(
          children: <Widget>[row1, row2, AddemergencycontactForm()],
        ),
      ))),
    );
  }
}
