import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'Screens/viewreminders.dart';

class Addmedicine extends StatefulWidget {
  @override
  _AddmedicineState createState() => _AddmedicineState();
}

class _AddmedicineState extends State<Addmedicine> {
  List<bool> _data = [false, false, false, false, false, false, false];
  List<String> _days = ['sat', 'sun', 'mon', 'tue', 'wed', 'thur', 'fri'];
  @override
  final _formKey = GlobalKey<FormState>();
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Add Reminder'),
        backgroundColor: const Color(0xff68b2a0),
      ),
      backgroundColor: const Color(0xffe0ecde),
      body: SafeArea(
        child: Container(
          child: new SingleChildScrollView(
              child: Column(children: [
            Center(
              child: Text(
                'New Reminder',
                style: TextStyle(
                  fontFamily: 'Montserrat-Bold',
                  fontSize: 13,
                  color: const Color(0xff205072),
                  height: 1.3846153846153846,
                ),
                textAlign: TextAlign.center,
              ),
            ),
            Center(
              child: Text(
                'Enter your reminder details',
                style: TextStyle(
                  fontFamily: 'Montserrat-Medium',
                  fontSize: 10,
                  color: const Color(0xff68b2a0),
                  height: 1.5,
                ),
                textAlign: TextAlign.center,
              ),
            ),
            Home(),
            Container(
              margin:
                  new EdgeInsets.symmetric(vertical: 20.0, horizontal: 10.0),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                color: const Color(0xffffffff),
              ),
              child: TextFormField(
                autofocus: true,
                style: TextStyle(color: Colors.green),
                decoration: InputDecoration(
                    prefixIcon: Icon(Icons.tab_outlined),
                    labelText: 'Name',
                    labelStyle: TextStyle(color: Colors.green),
                    focusedBorder: OutlineInputBorder(),
                    enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.green),
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
            Container(
              margin:
                  new EdgeInsets.symmetric(vertical: 20.0, horizontal: 10.0),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                color: const Color(0xffffffff),
              ),
              child: TextFormField(
                autofocus: true,
                style: TextStyle(color: Colors.green),
                decoration: InputDecoration(
                    prefixIcon: Icon(Icons.tab_outlined),
                    labelText: 'Times per day',
                    labelStyle: TextStyle(color: Colors.green),
                    focusedBorder: OutlineInputBorder(),
                    enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.green),
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
            _buildCheckBoxes(),
            Container(
              margin:
                  new EdgeInsets.symmetric(vertical: 20.0, horizontal: 10.0),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                color: const Color(0xffffffff),
              ),
              child: TextFormField(
                autofocus: true,
                style: TextStyle(color: Colors.green),
                decoration: InputDecoration(
                    prefixIcon: Icon(Icons.tab_outlined),
                    labelText: 'Time',
                    labelStyle: TextStyle(color: Colors.green),
                    focusedBorder: OutlineInputBorder(),
                    enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.green),
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
            Container(
              margin:
                  new EdgeInsets.symmetric(vertical: 20.0, horizontal: 10.0),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                color: const Color(0xffffffff),
              ),
              child: TextFormField(
                autofocus: true,
                style: TextStyle(color: Colors.green),
                decoration: InputDecoration(
                    prefixIcon: Icon(Icons.tab_outlined),
                    labelText: 'Course Duration',
                    labelStyle: TextStyle(color: Colors.green),
                    focusedBorder: OutlineInputBorder(),
                    enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: Colors.green),
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
            Container(
              margin:
                  new EdgeInsets.symmetric(vertical: 20.0, horizontal: 10.0),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(21.0),
                gradient: RadialGradient(
                  center: Alignment(-0.88, -1.0),
                  radius: 1.35,
                  colors: [const Color(0xff7be495), const Color(0xff329d9c)],
                  stops: [0.0, 1.0],
                ),
              ),
              child: FlatButton(
                child: Text('Save'),
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => ViewReminders()),
                  );
                },
              ),
            )
          ])),
        ),
      ),
    );
  }

  Widget _buildCheckBoxes() {
    List<Widget> list = new List();
    Widget cb;

    for (int i = 0; i < _data.length; i++) {
      cb = Column(
        children: [
          Text(
            _days[i],
            style: TextStyle(
              fontFamily: 'Montserrat-Bold',
              fontSize: 13,
              color: Colors.green,
              height: 1.3846153846153846,
            ),
            textAlign: TextAlign.center,
          ),
          Container(
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
                  offset: Offset(34, 34),
                  blurRadius: 89,
                ),
              ],
            ),
            child: Container(
              child: Checkbox(
                activeColor: Colors.white,
                checkColor: Colors.green,
                value: _data[i],
                onChanged: (bool value) {
                  setState(() {
                    // need to update _data[?]
                    _data[i] = value;
                  });
                },
              ),
            ),
          ),
        ],
      );
      list.add(cb);
    }

    return Row(mainAxisAlignment: MainAxisAlignment.center, children: list);
  }
}

class ListItem {
  int value;
  String name;

  ListItem(this.value, this.name);
}

class Home extends StatefulWidget {
  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  List<ListItem> _dropdownItems = [
    ListItem(1, "Medicine"),
    ListItem(2, "Syringe"),
    ListItem(3, "Check Up"),
  ];

  List<DropdownMenuItem<ListItem>> _dropdownMenuItems;
  ListItem _selectedItem;

  void initState() {
    super.initState();
    _dropdownMenuItems = buildDropDownMenuItems(_dropdownItems);
    _selectedItem = _dropdownMenuItems[0].value;
  }

  List<DropdownMenuItem<ListItem>> buildDropDownMenuItems(List listItems) {
    List<DropdownMenuItem<ListItem>> items = List();
    for (ListItem listItem in listItems) {
      items.add(
        DropdownMenuItem(
          child: Text(
            listItem.name,
            style: TextStyle(
              fontFamily: 'Montserrat-Bold',
              fontSize: 15,
              color: const Color(0xff68b2a0),
              height: 1.1333333333333333,
            ),
            textAlign: TextAlign.left,
          ),
          value: listItem,
        ),
      );
    }
    return items;
  }

  @override
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Container(
        padding: const EdgeInsets.only(left: 10.0, right: 10.0),
        decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(10.0),
            color: Color(0xffffffff),
            border: Border.all(color: Colors.green)),
        child: DropdownButtonHideUnderline(
          child: DropdownButton(
              value: _selectedItem,
              items: _dropdownMenuItems,
              onChanged: (value) {
                setState(() {
                  _selectedItem = value;
                });
              }),
        ),
      ),
    );
  }
}
