import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import '../widgets/Editprofile/editform.dart';

class EditProfilePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    var row1 = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Padding(padding: EdgeInsets.only(top: 50.0)),
        Text(
          'Edit Profile',
          style: TextStyle(fontSize: 30, fontWeight: FontWeight.bold),
        ),
      ],
    );
    var row2 = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          'Enter your new account details to edit your account',
          style: TextStyle(fontSize: 15, color: Colors.green),
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
          children: <Widget>[row1, row2, EditProfileForm()],
        ),
      ))),
    );
  }
}
