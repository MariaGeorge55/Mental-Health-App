import 'dart:ffi';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:project_mobile/Home.dart';
//import 'package:project_mobile/Signupold.dart';
import 'package:phone_number/phone_number.dart';
import 'package:project_mobile/users.dart';

const pattern = r'(^(?:[+0]9)?[0-9]{10,12}$)';
final validatePhone = RegExp(pattern);

class SignupPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Container(
      decoration: new BoxDecoration(
        gradient: new LinearGradient(
            colors: [
              Color.fromRGBO(9, 6, 116, 1.0),
              Color.fromRGBO(18, 11, 232, 1.0)
            ],
            begin: const FractionalOffset(0.5, 0.8),
            end: const FractionalOffset(0.0, 0.5),
            stops: [0.0, 5.0],
            tileMode: TileMode.clamp),
      ),
      child: Column(
        children: <Widget>[Header(), SignUpForm()],
      ),
    ));
  }
}

class Header extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
        margin: EdgeInsets.only(top: 0.0),
        child: Column(
          children: [
            Padding(padding: EdgeInsets.only(top: 50.0)),
            Text(
              'Sign Up',
              style: TextStyle(
                fontFamily: 'Lato-Bold',
                fontSize: 21,
                color: const Color(0xffffffff),
              ),
            ),
            Text(
              'Enter your details to create your account',
              style: TextStyle(
                  fontSize: 15, color: Color.fromRGBO(176, 208, 255, 1)),
            ),
          ],
        ));
  }
}

class SignUpForm extends StatefulWidget {
  @override
  SignUpFormState createState() {
    return SignUpFormState();
  }
}

class SignUpFormState extends State<SignUpForm> {
  var _passwordVisible;
  @override
  void initState() {
    _passwordVisible = false;
    super.initState();
  }

  // Create a global key that uniquely identifies the Form widget
  // and allows validation of the form.
  //
  // Note: This is a GlobalKey<FormState>,
  // not a GlobalKey<MyCustomFormState>.
  final _formKey = GlobalKey<FormState>();
  static final validCharacters = RegExp(r'^[a-zA-Z0-9]+$');

  static final validCharactersPassword = RegExp(r'^[a-zA-Z0-9_\-=@\.;]+$');
  static final validateDate = RegExp("[0-9/]");
  Users u = new Users();
  final usernamecontroller = TextEditingController();
  final passwordcontroller = TextEditingController();
  final emergencycontactnamecontroller = TextEditingController();
  final datecontroller = TextEditingController();
  final emergencycontactcontroller = TextEditingController();
  final emailcontroller = TextEditingController();
  @override
  Widget build(BuildContext context) {
    Widget okButton = FlatButton(
      child: Text(
        "Ok",
      ),
      onPressed: () {
        Navigator.of(context, rootNavigator: true).pop();
        Navigator.of(context).push(new MaterialPageRoute(
            builder: (context) => HomePage(
                u: u.signup(
                    usernamecontroller.text,
                    passwordcontroller.text,
                    emergencycontactnamecontroller.text,
                    datecontroller.text,
                    emailcontroller.text,
                    emergencycontactcontroller.text))));
      },
    );
    // Build a Form widget using the _formKey created above.
    return Form(
      key: _formKey,
      child: Container(
        width: 200,
        margin: EdgeInsets.only(left: 100, right: 110),
        child: Column(
          children: <Widget>[
            Padding(padding: EdgeInsets.only(top: 20.0)),
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                color: const Color(0xffffffff),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0x21329d9c),
                    offset: Offset(0, 13),
                    blurRadius: 34,
                  ),
                ],
              ),
              child: TextFormField(
                controller: usernamecontroller,
                style: TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                decoration: InputDecoration(
                    prefixIcon: Icon(Icons.perm_identity),
                    labelText: 'Enter your Name',
                    labelStyle:
                        TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                    focusedBorder: OutlineInputBorder(
                        borderSide:
                            BorderSide(color: Color.fromRGBO(32, 80, 114, 1))),
                    enabledBorder: OutlineInputBorder(
                      borderSide:
                          BorderSide(color: Color.fromRGBO(32, 80, 114, 1)),
                      borderRadius: BorderRadius.circular(21.0),
                    )),
                validator: (value) {
                  if (value.isEmpty) {
                    return 'Please enter some text';
                  }
                  if (!value.contains(validCharacters)) {
                    return 'Please enter the correct format';
                  }
                  return null;
                },
              ),
            ),
            Padding(padding: EdgeInsets.only(top: 10.0)),
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                color: const Color(0xffffffff),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0x21329d9c),
                    offset: Offset(0, 13),
                    blurRadius: 34,
                  ),
                ],
              ),
              child: TextFormField(
                controller: datecontroller,
                style: TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                decoration: InputDecoration(
                    prefixIcon: Icon(Icons.date_range),
                    labelText: 'DD/MM/YY',
                    labelStyle:
                        TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                    focusedBorder: OutlineInputBorder(
                        borderSide:
                            BorderSide(color: Color.fromRGBO(32, 80, 114, 1))),
                    enabledBorder: OutlineInputBorder(
                      borderSide:
                          BorderSide(color: Color.fromRGBO(32, 80, 114, 1)),
                      borderRadius: BorderRadius.circular(21.0),
                    )),
                validator: (value) {
                  if (value.isEmpty) {
                    return 'Please enter some text';
                  }
                  if (!value.contains(validateDate)) {
                    return 'Please enter the correct format';
                  }
                  return null;
                },
              ),
            ),
            Padding(padding: EdgeInsets.only(top: 10.0)),
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                color: const Color(0xffffffff),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0x21329d9c),
                    offset: Offset(0, 13),
                    blurRadius: 34,
                  ),
                ],
              ),
              child: TextFormField(
                controller: emergencycontactnamecontroller,
                style: TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                decoration: InputDecoration(
                    prefixIcon: Icon(Icons.perm_identity),
                    labelText: 'Emergency Contact name',
                    labelStyle:
                        TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                    focusedBorder: OutlineInputBorder(
                        borderSide:
                            BorderSide(color: Color.fromRGBO(32, 80, 114, 1))),
                    enabledBorder: OutlineInputBorder(
                      borderSide:
                          BorderSide(color: Color.fromRGBO(32, 80, 114, 1)),
                      borderRadius: BorderRadius.circular(21.0),
                    )),
                validator: (value) {
                  if (value.isEmpty) {
                    return 'Please enter some text';
                  }
                  if (!value.contains(validCharacters)) {
                    return 'Please enter the correct format';
                  }
                  return null;
                },
              ),
            ),
            Padding(padding: EdgeInsets.only(top: 10.0)),
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                color: const Color(0xffffffff),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0x21329d9c),
                    offset: Offset(0, 13),
                    blurRadius: 34,
                  ),
                ],
              ),
              child: TextFormField(
                controller: emergencycontactcontroller,
                style: TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                decoration: InputDecoration(
                    prefixIcon: Icon(Icons.add_alert),
                    labelText: 'Emergency Contact Number',
                    labelStyle:
                        TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                    focusedBorder: OutlineInputBorder(
                        borderSide:
                            BorderSide(color: Color.fromRGBO(32, 80, 114, 1))),
                    enabledBorder: OutlineInputBorder(
                      borderSide:
                          BorderSide(color: Color.fromRGBO(32, 80, 114, 1)),
                      borderRadius: BorderRadius.circular(21.0),
                    )),
                validator: (value) {
                  if (value.isEmpty) {
                    return 'Please enter some text';
                  }
                  if (!value.contains(validatePhone)) {
                    return 'Please enter the correct format';
                  }
                  return null;
                },
              ),
            ),
            Padding(padding: EdgeInsets.only(top: 10.0)),
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                color: const Color(0xffffffff),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0x21329d9c),
                    offset: Offset(0, 13),
                    blurRadius: 34,
                  ),
                ],
              ),
              child: TextFormField(
                controller: emailcontroller,
                style: TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                decoration: InputDecoration(
                    prefixIcon: Icon(Icons.email),
                    labelText: 'Email',
                    labelStyle:
                        TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                    focusedBorder: OutlineInputBorder(
                        borderSide:
                            BorderSide(color: Color.fromRGBO(32, 80, 114, 1))),
                    enabledBorder: OutlineInputBorder(
                      borderSide:
                          BorderSide(color: Color.fromRGBO(32, 80, 114, 1)),
                      borderRadius: BorderRadius.circular(21.0),
                    )),
                validator: (value) {
                  if (value.isEmpty) {
                    return 'Please enter some text';
                  }
                  if (!value.contains('.com')) {
                    return 'Please enter the correct email format';
                  }
                  return value.contains('@')
                      ? null
                      : 'Please enter the correct email format';
                },
              ),
            ),
            Padding(padding: EdgeInsets.only(top: 10.0)),
            Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                color: const Color(0xffffffff),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0x21329d9c),
                    offset: Offset(0, 13),
                    blurRadius: 34,
                  ),
                ],
              ),
              child: TextFormField(
                controller: passwordcontroller,
                obscureText: !_passwordVisible,
                style: TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                decoration: InputDecoration(
                    prefixIcon: IconButton(
                      icon: Icon(
                        _passwordVisible
                            ? Icons.visibility
                            : Icons.visibility_off,
                        color: Theme.of(context).primaryColorDark,
                      ),
                      onPressed: () {
                        // Update the state i.e. toogle the state of passwordVisible variable
                        setState(() {
                          _passwordVisible = !_passwordVisible;
                        });
                      },
                    ),
                    labelText: 'Password',
                    labelStyle:
                        TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
                    focusedBorder: OutlineInputBorder(
                        borderSide:
                            BorderSide(color: Color.fromRGBO(32, 80, 114, 1))),
                    enabledBorder: OutlineInputBorder(
                      borderSide:
                          BorderSide(color: Color.fromRGBO(32, 80, 114, 1)),
                      borderRadius: BorderRadius.circular(21.0),
                    )),
                validator: (value) {
                  if (value.isEmpty) {
                    return 'Please enter some text';
                  }
                  if (!value.contains(validCharactersPassword)) {
                    return 'Please enter the correct format';
                  }
                  ;
                  return null;
                },
              ),
            ),
            Padding(padding: const EdgeInsets.symmetric(vertical: 5.0)),
            Container(
              decoration: BoxDecoration(
                gradient: new LinearGradient(
                    colors: [
                      Color.fromRGBO(18, 11, 232, 1.0),
                      Color.fromRGBO(107, 164, 249, 1.0)
                    ],
                    begin: const FractionalOffset(0.1, 1.0),
                    end: const FractionalOffset(0.8, 0.5),
                    stops: [0.0, 1.0],
                    tileMode: TileMode.clamp),
                boxShadow: [
                  BoxShadow(
                      color: const Color(0xff000000),
                      offset: Offset(0, 13),
                      blurRadius: 15,
                      spreadRadius: -10.0),
                ],
                border: Border.all(
                  color: Colors.black,
                ),
                borderRadius: BorderRadius.all(Radius.circular(30)),
              ),
              child: Align(
                alignment: Alignment.topCenter,
                child: FlatButton(
                  onPressed: () {
                    // Validate returns true if the form is valid, or false
                    // otherwise.

                    if (_formKey.currentState.validate()) {
                      return showDialog(
                        context: context,
                        builder: (context) {
                          return AlertDialog(
                            content: Text("Account created successfully"),
                            actions: [okButton],
                          );
                        },
                      );
                    }
                  },
                  child: Text('Sign Up',
                      style: TextStyle(
                        fontFamily: 'Montserrat-Bold',
                        fontSize: 13,
                        color: const Color(0xffffffff),
                      )),
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}
