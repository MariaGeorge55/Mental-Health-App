import 'package:flutter/material.dart';
import 'viewjournals.dart';

class Journal extends StatefulWidget {
  @override
  _Journal createState() => _Journal();
}

class _Journal extends State<Journal> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Container(
          decoration: new BoxDecoration(
            gradient: new LinearGradient(
                colors: [
                  Color.fromRGBO(9, 6, 116, 1.0),
                  Color.fromRGBO(18, 11, 232, 1.0)
                ],
                begin: const FractionalOffset(0.5, 0.0),
                end: const FractionalOffset(0.0, 0.5),
                stops: [0.0, 1.0],
                tileMode: TileMode.clamp),
          ),
          child: Container(
            child: Column(children: [
              TitleText(),
              LoginForm(),
            ]),
          ),
        ),
      ),
    );
  }
}

class TitleText extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 100,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.start,
        children: [
          Text(
            '\n New Journal:',
            style: TextStyle(
              fontFamily: 'Montserrat-Bold',
              fontSize: 21,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}

class LoginForm extends StatefulWidget {
  @override
  LoginFormState createState() {
    return LoginFormState();
  }
}

class LoginFormState extends State<LoginForm> {
  // Create a global key that uniquely identifies the Form widget
  // and allows validation of the form.
  //
  // Note: This is a GlobalKey<FormState>,
  // not a GlobalKey<MyCustomFormState>.
  final _formKey = GlobalKey<FormState>();
  @override
  Widget build(BuildContext context) {
    // Build a Form widget using the _formKey created above.
    return Form(
      key: _formKey,
      child: Column(
        children: <Widget>[
          Padding(padding: EdgeInsets.fromLTRB(30, 0, 30, 0)),
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
            child: Container(
              height: 450,
              width: 370,
              child: TextFormField(
                style: TextStyle(color: Colors.green),
                decoration: InputDecoration(
                    contentPadding: const EdgeInsets.symmetric(vertical: 300.0),
                    labelText: 'Enter diary',
                    labelStyle: TextStyle(color: Colors.black),
                    focusedBorder: OutlineInputBorder(
                        borderSide: BorderSide(color: Colors.green)),
                    enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.black),
                      borderRadius: BorderRadius.circular(21.0),
                    )),
                validator: (value) {
                  if (value.isEmpty) {
                    return 'Please enter some text';
                  }
                  return null;
                },
              ),
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
          ),
          Padding(padding: const EdgeInsets.only(left: 500.0)),
          Container(
            width: 200,
            child: Container(
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
              child: FlatButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => ViewJournals()),
                  );
                },
                child: Text("Save",
                    style: TextStyle(
                      fontFamily: 'Montserrat-Bold',
                      fontSize: 13,
                      color: const Color(0xffffffff),
                    )),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
