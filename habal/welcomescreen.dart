import 'package:flutter/material.dart';
import 'Signup.dart';
import 'Login.dart';

class WelcomeScreen extends StatefulWidget {
  @override
  _WelcomeScreenState createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      drawer: Drawer(
        child: ListView(
          children: <Widget>[
            ListTile(
              title: Text('Home'),
              onTap: () => Navigator.pushNamed(context, 'home'),
            ),
            ListTile(
              title: Text('Profile'),
              onTap: () => Navigator.pushNamed(context, 'profile'),
            ),
            ListTile(
              title: Text('Add Journal'),
              onTap: () => Navigator.pushNamed(context, 'addjournal'),
            ),
            ListTile(
              title: Text('Reminders'),
              onTap: () => Navigator.pushNamed(context, 'reminders'),
            ),
            ListTile(
              title: Text('Contact Us'),
              onTap: () => Navigator.pushNamed(context, 'Contact Us'),
            ),
          ],
        ),
      ),
      appBar: AppBar(
        title: Text('Welcome Page'),
        backgroundColor: const Color(0xff68b2a0),
      ),
      backgroundColor: const Color(0xffffffff),
      body: SafeArea(
        child: Column(children: [
          Himage(),
          TitleText(),
          SubtitleText(),
          Iimage(),
          Signinbutton(),
          Signuptext()
        ]),
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
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            'This is HealthTracker,\n           Welcome!',
            style: TextStyle(
              fontFamily: 'Montserrat-Bold',
              fontSize: 21,
              color: const Color(0xff205072),
            ),
          ),
        ],
      ),
    );
  }
}

class SubtitleText extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          'Track your medications and get reminders,\npersonal diary, general health instructions',
          style: TextStyle(
            fontFamily: 'Montserrat-Medium',
            fontSize: 13,
            color: const Color(0xff68b2a0),
          ),
        ),
      ],
    );
  }
}

class Iimage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.fromLTRB(50, 20, 0, 30),
      child: Row(
        children: [
          Image.asset(
            'images/Doctor.png',
            width: 310,
            height: 250,
          )
        ],
      ),
    );
  }
}

class Himage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.fromLTRB(0, 30, 0, 0),
      child: Row(
        children: [
          Image.asset(
            'images/icon.png',
            width: 400,
            height: 40,
          )
        ],
      ),
    );
  }
}

class Signinbutton extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          margin: EdgeInsets.fromLTRB(120, 0, 10, 0),
          width: 180,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(21.0),
            gradient: RadialGradient(
              center: Alignment(-0.88, -1.0),
              radius: 1.35,
              colors: [const Color(0xff7be495), const Color(0xff329d9c)],
              stops: [0.0, 1.0],
            ),
            boxShadow: [
              BoxShadow(
                color: const Color(0x36329d9c),
                offset: Offset(15, 15),
                blurRadius: 40,
              ),
            ],
          ),
          child: FlatButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => LoginPage()),
              );
            },
            child: Text(
              "Log in",
            ),
          ),
        )
      ],
    );
  }
}

class Signuptext extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.fromLTRB(30, 0, 0, 0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            'Do not have an account?',
            style: TextStyle(
              fontFamily: 'Montserrat-Medium',
              fontSize: 12,
              color: const Color(0xff68b2a0),
            ),
          ),
          FlatButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => SignupPage()),
              );
            },
            child: Text(
              "Sign Up",
              style: TextStyle(
                fontFamily: 'Montserrat-Medium',
                fontSize: 12,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
