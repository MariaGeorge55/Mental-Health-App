import 'dart:ffi';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:project_mobile/Home.dart';
import 'package:project_mobile/Signup.dart';
import 'package:project_mobile/users.dart';

class LoginPage extends StatelessWidget {
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
        child: Center(
          child: Column(
            children: <Widget>[Header(), LoginForm()],
          ),
        ),
      ),
    );
  }
}

class Header extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
        margin: EdgeInsets.only(top: 200.0),
        child: Column(
          children: [
            Padding(padding: EdgeInsets.only(top: 50.0)),
            Text(
              'Login',
              style: TextStyle(
                fontFamily: 'Lato-Bold',
                fontSize: 21,
                color: const Color(0xffffffff),
              ),
            ),
            Text(
              'Enter your login details to access your account',
              style: TextStyle(
                  fontSize: 15, color: Color.fromRGBO(176, 208, 255, 1)),
            ),
          ],
        ));
  }
}

class LoginForm extends StatefulWidget {
  @override
  LoginFormState createState() {
    return LoginFormState();
  }
}

class LoginFormState extends State<LoginForm> {
  FocusNode myFocusNode;
  var _passwordVisible;
  @override
  void initState() {
    _passwordVisible = false;
    super.initState();
    myFocusNode = FocusNode();
  }

  // Create a global key that uniquely identifies the Form widget
  // and allows validation of the form.
  //
  // Note: This is a GlobalKey<FormState>,
  // not a GlobalKey<MyCustomFormState>.
  final _formKey = GlobalKey<FormState>();
  static final validCharacters = RegExp(r'^[a-zA-Z0-9]+$');
  static final validCharactersPassword = RegExp(r'^[a-zA-Z0-9_\-=@\.;]+$');
  Users users = new Users();
  final usernamecontroller = TextEditingController();
  final passwordcontroller = TextEditingController();
  bool checkuserexist() {
    User u =
        users.validatelogin(usernamecontroller.text, passwordcontroller.text);
    if (u != null) {
      return true;
    } else {
      return false;
    }
  }

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
                u: users.validatelogin(
                    usernamecontroller.text, passwordcontroller.text))));
      },
    );

    Widget okButton2 = FlatButton(
      child: Text(
        "login again",
      ),
      onPressed: () {
        Navigator.of(context, rootNavigator: true).pop();
        Navigator.of(context)
            .push(new MaterialPageRoute(builder: (context) => LoginPage()));
      },
    );
    // Build a Form widget using the _formKey created above.
    return Form(
      key: _formKey,
      child: Column(
        children: <Widget>[
          Padding(padding: EdgeInsets.only(top: 20.0)),
          Container(
            width: 200,
            height: 80,
            decoration: BoxDecoration(
              color: const Color(0xffffffff),
              boxShadow: [
                BoxShadow(
                  color: const Color(0x21329d9c),
                  offset: Offset(0, 13),
                  blurRadius: 34,
                ),
              ],
              border: Border.all(
                color: Colors.black,
              ),
              borderRadius: BorderRadius.all(Radius.circular(30)),
            ),
            child: TextFormField(
              controller: usernamecontroller,
              autofocus: true,
              style: TextStyle(
                  fontFamily: "Montserrat-Medium",
                  color: Color.fromRGBO(32, 80, 114, 1)),
              decoration: InputDecoration(
                prefixIcon: Icon(Icons.perm_identity),
                labelText: 'Enter your username',
                labelStyle: TextStyle(color: Color.fromRGBO(32, 80, 114, 1)),
              ),
              validator: (value) {
                if (value.isEmpty) {
                  return 'Please enter some text';
                }
                if (!value.contains(validCharacters)) {
                  return 'Please enter the correct format';
                }
                ;
                return null;
              },
            ),
          ),
          Padding(padding: EdgeInsets.only(top: 20.0)),
          Container(
            width: 200,
            height: 80,
            decoration: BoxDecoration(
              color: const Color(0xffffffff),
              boxShadow: [
                BoxShadow(
                  color: const Color(0x21329d9c),
                  offset: Offset(0, 13),
                  blurRadius: 34,
                ),
              ],
              border: Border.all(
                color: Colors.red,
              ),
              borderRadius: BorderRadius.all(Radius.circular(30)),
            ),
            child: TextFormField(
              controller: passwordcontroller,
              focusNode: myFocusNode,
              obscureText: !_passwordVisible,
              style: TextStyle(
                  fontFamily: "Montserrat-Medium",
                  color: Color.fromRGBO(32, 80, 114, 1)),
              decoration: InputDecoration(
                prefixIcon: IconButton(
                  icon: Icon(
                    _passwordVisible ? Icons.visibility : Icons.visibility_off,
                    color: Theme.of(context).primaryColorDark,
                  ),
                  onPressed: () {
                    // Update the state i.e. toogle the state of passwordVisible variable
                    setState(() {
                      _passwordVisible = !_passwordVisible;
                    });
                  },
                ),
                labelText: 'Enter your password',
                labelStyle: TextStyle(
                    fontFamily: "Montserrat-Medium",
                    color: Color.fromRGBO(32, 80, 114, 1)),
              ),
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
            width: 180,
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
                child: Text('Login'),
                onPressed: () {
                  // Validate returns true if the form is valid, or false
                  // otherwise.
                  if (_formKey.currentState.validate()) {
                    if (checkuserexist() == true) {
                      return showDialog(
                        context: context,
                        builder: (context) {
                          return AlertDialog(
                            content: Text("Logged in successfully"),
                            actions: [okButton],
                          );
                        },
                      );
                    } else {
                      return showDialog(
                        context: context,
                        builder: (context) {
                          return AlertDialog(
                            content: Text("wrong username or password"),
                            actions: [okButton2],
                          );
                        },
                      );
                    }
                    // If the form is valid, display a Snackbar.

                  }
                },
              ),
            ),
          )
        ],
      ),
    );
  }
}
