using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Onclick : MonoBehaviour {

    private void OnMouseDown() ////this function is called when the user has pressed the mouse button while over the Collider.
    {
        // Destroy(gameObject);
        SceneManager.LoadScene(1);
    }
}
