import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:project_mobile/EmotionAnalysisPic.dart';
import 'package:project_mobile/viewjournals.dart';
import 'viewjournals.dart';
import 'package:project_mobile/journal.dart';
import 'welcomescreen (1).dart';
import 'Login.dart';
import 'contactus.dart';
import 'journal.dart';
import 'Home.dart';
import 'profile.dart';
import 'users.dart';
import 'viewreminders.dart';
import 'viewjournals.dart';
import 'searchfaq.dart';
import 'searchmedicine.dart';

class HomePage extends StatelessWidget {
  final User u;
  HomePage({this.u});

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
    return MaterialApp(
      home: Scaffold(
          drawer: Drawer(
            child: ListView(
              children: <Widget>[
                ListTile(
                    title: Text('Home'),
                    onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => HomePage(u: this.u)),
                        )),
                ListTile(
                    title: Text('Profile'),
                    onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => ProfilePage()),
                        )),
                /* ListTile(
                    title: Text('Add Journal'),
                    onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(builder: (context) => Journal()),
                        )),
                ListTile(
                    title: Text('My Reminders'),
                    onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => ViewReminders()),
                        )),
                ListTile(
                    title: Text('Search for medicine'),
                    onTap: () => Navigator.push(
                          context,
                          MaterialPageRoute(builder: (context) => SearchMed()),
                        )),*/
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
            title: Text('Home Page'),
            backgroundColor: Colors.indigo,
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
                  child: FlatButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => EmotionAnalysisPic()),
                      );
                    },
                    child: Text("Analyze",
                        style: TextStyle(
                          fontFamily: 'Montserrat-Bold',
                          fontSize: 13,
                          color: const Color(0xffffffff),
                        )),
                  ),
                ),
              ],
            ).toList()),
          ))),
    );
  }
}
