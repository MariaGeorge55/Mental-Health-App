import 'package:firebase_auth/firebase_auth.dart';

Future<bool> signIn(String email, String password) async {
  try {
    await FirebaseAuth.instance
        .signInWithEmailAndPassword(email: email, password: password);
    print(FirebaseAuth.instance.currentUser.uid);
    return true;
  } catch (e) {
    print("Exeption in Sign in" + e);
    return false;
  }
}

Future<bool> register(String email, String password) async {
  try {
    print("fire Emaillll email" + email);
    await FirebaseAuth.instance
        .createUserWithEmailAndPassword(email: email, password: password);
    print("fire Emaillll email" + email);

    print("fire auth");
    print(FirebaseAuth.instance.currentUser.uid);
    return true;
  } on FirebaseAuthException catch (e) {
    print("fire eerr");
    print(e);
    if (e.code == 'weak-password') {
      print('The password provided is too weak.');
    } else if (e.code == 'email-already-in-use') {
      print('The account already exists for that email.');
    }
    return false;
  } catch (e) {
    print("fire auth   catch");
    print(e.toString());
    return false;
  }
}
