import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:project_mobile/Screens/viewjournals.dart';
import 'Screens/viewjournals.dart';
import 'Screens/contactus.dart';
import 'Screens/profile.dart';
import 'models/users.dart';
import 'Screens/viewreminders.dart';
import 'Screens/viewjournals.dart';
import 'Screens/searchfaq.dart';

class HomePage extends StatelessWidget {
  User u = new User();
  HomePage.ut({this.u});
  HomePage();
  @override
  Widget build(BuildContext context) {
    var Row1 = Row(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Flexible(
            fit: FlexFit.tight,
            flex: 2,
            child: Container(
                width: 385.1,
                height: 250.2,
                decoration: BoxDecoration(
                  /*image: new DecorationImage(
                    image: new AssetImage("images/Avatar 3.png"),
                  ),*/
                  borderRadius: BorderRadius.only(
                    bottomRight: Radius.circular(34.0),
                    bottomLeft: Radius.circular(34.0),
                  ),
                  color: const Color(0xffe0ecde),
                ),
                padding: new EdgeInsets.only(top: 30),
                child: Text(
                  '      Hello ${u.username}',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ))),
        Padding(padding: EdgeInsets.only(top: 50.0)),
      ],
    );
    var Row2 = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Image(image: AssetImage('images/Icons- brain.png')),
        Padding(padding: EdgeInsets.only(right: 7.0)),
        FlatButton(
          child: Text(
            'Reminders',
            style: TextStyle(fontSize: 15, color: Colors.black),
          ),
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => ViewReminders()),
            );
          },
        ),
      ],
    );
    var Row3 = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Padding(padding: EdgeInsets.only(top: 5.0)),
        Image(image: AssetImage('images/Icons- path.png')),
        Padding(padding: EdgeInsets.only(right: 7.0)),
        FlatButton(
          child: Text(
            'Journals',
            style: TextStyle(fontSize: 15, color: Colors.black),
          ),
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => ViewJournals()),
            );
          },
        ),
      ],
    );
    var Row4 = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Padding(padding: EdgeInsets.only(top: 5.0)),
        Image(image: AssetImage('images/Icons- lab.png')),
        Padding(padding: EdgeInsets.only(right: 7.0)),
        Text(
          'Analysis',
          style: TextStyle(fontSize: 15, color: Colors.black),
        ),
      ],
    );
    return Scaffold(
        drawer: Drawer(
          child: ListView(
            children: <Widget>[
              ListTile(
                  title: Text('Home'),
                  onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => HomePage.ut(u: this.u)),
                      )),
              ListTile(
                  title: Text('Profile'),
                  onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => ProfilePage()),
                      )),
              ListTile(
                  title: Text('FAQ'),
                  onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => SearchFaq()),
                      )),
              ListTile(
                  title: Text('Contact Us'),
                  onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => ContactUs()),
                      )),
            ],
          ),
        ),
        appBar: AppBar(
          title: Row(
            children: [
              Text('Home Page'),
              Container(
                margin: EdgeInsets.fromLTRB(100, 0, 0, 0),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(21.0),
                  color: Colors.red,
                  boxShadow: [
                    BoxShadow(
                        color: const Color(0x36329d9c),
                        offset: Offset(15, 15),
                        blurRadius: 40,
                        spreadRadius: 20),
                  ],
                ),
                child: FlatButton(
                  child: Text(
                    "SOS",
                  ),
                ),
              )
            ],
          ),
          backgroundColor: const Color(0xff68b2a0),
//            title: Center(child: Image(image: AssetImage('images/Icon.png'))),
        ),
        body: Center(
            child: Container(
          child: ListView(
              children: ListTile.divideTiles(
            context: context,
            tiles: [
              ListTile(
                title: Row1,
              ),
              Divider(),
              ListTile(
                title: Row2,
              ),
              ListTile(
                title: Row3,
              ),
              Divider(),
              ListTile(
                title: Row4,
              ),
              Divider(),
            ],
          ).toList()),
        )));
  }
}
