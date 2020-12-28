import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'journal.dart';

class ViewJournals extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: new LinearGradient(
              colors: [
                Color.fromRGBO(9, 6, 116, 1.0),
                Color.fromRGBO(18, 11, 232, 1.0)
              ],
              begin: const FractionalOffset(0.1, 1.0),
              end: const FractionalOffset(0.8, 0.5),
              stops: [0.0, 1.0],
              tileMode: TileMode.clamp),
        ),
        child: Column(
          children: [
            Container(
              width: 850,
              height: 70,
              margin: EdgeInsets.only(left: 30.0, top: 50.0, bottom: 0.0),
              child: Row(children: [
                Text(
                  'Journals:',
                  style: TextStyle(
                      fontSize: 25,
                      color: Colors.white,
                      fontWeight: FontWeight.bold),
                ),
                Padding(
                  padding: const EdgeInsets.only(
                    left: 120,
                    top: 10.0,
                    bottom: 0.0,
                  ),
                  child: FlatButton(
                    child: Text(
                      '+ Journals',
                      style: TextStyle(
                          fontSize: 20,
                          color: Colors.white,
                          fontWeight: FontWeight.bold),
                    ),
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => Journal()),
                      );
                    },
                  ),
                )
              ]),
            ),
            Center(
                child: new SingleChildScrollView(
              child: Container(
                  width: 385.1,
                  height: 470,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(34.0),
                      topRight: Radius.circular(34.0),
                      bottomRight: Radius.circular(34.0),
                      bottomLeft: Radius.circular(34.0),
                    ),
                    color: Colors.white,
                  ),
                  child: ListView.builder(
                    itemCount: 4,
                    itemBuilder: (context, index) {
                      return ListTile(
                        leading: Image(image: AssetImage('images/Path.png')),
                        title: Column(children: [
                          Padding(
                            padding: const EdgeInsets.fromLTRB(0, 0, 150, 0),
                            child: FlatButton(
                              child: Text('About Today'),
                              onPressed: () {},
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.fromLTRB(0, 0, 120, 0),
                            child: Text(
                              'added 2 months ago',
                              style: TextStyle(
                                  fontSize: 14,
                                  color: Colors.grey,
                                  fontWeight: FontWeight.normal),
                            ),
                          )
                        ]),
                      );
                    },
                  )),
            )),
          ],
        ),
      ),
    );
  }
}
