using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

public class UI : MonoBehaviour {

    public Text resText;
    public Text testText;
    public Text statusText;

    public RawImage rimg;

    public Classifier classifier;

    private Thread trainThread;

    // Use this for initialization
    void Start () {
        trainThread = new Thread(new ThreadStart(classifier.TrainEpoch));
    }

    void Update()
    {
        statusText.text = "STATUS:" + classifier.classifierState.ToString();
    }

    private bool IsProcessing()
    {
        if (classifier.classifierState == eClassifierState.TRAIN)
        {
            return true;
        }
        return false;
    }

    public void OnLoad()
    {
        if (IsProcessing())
        {
            return;
        }
        classifier.LoadModel();
    }

    public void OnSave()
    {
        if (IsProcessing())
        {
            return;
        }
        classifier.SaveModel();
    }

    public void OnTrain()
    {
        if (IsProcessing())
        {
            return;
        }
        trainThread.Start();
        Debug.Log("Epoch complete");
    }

    public void OnTest()
    {
        if (IsProcessing())
        {
            return;
        }
        float percent = classifier.TestAll();
        testText.text = "TEST: "+percent.ToString("N2")+"%";
    }

    public void OnClear()
    {
        classifier.painter.Clear();
    }

    public void OnPredect()
    {
        if (IsProcessing())
        {
            return;
        }
        classifier.ProcessImage();
    }

    public void OnApplicationQuit()
    {
        if (trainThread != null)
        {
            trainThread.Abort();
        }
    }
}
