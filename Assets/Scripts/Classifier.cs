using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using NuralNet;

public enum eDoodleCategory
{
    CAT = 0,
    RAINBOW = 1,
    TRAIN = 2
}

public class DoodleData
{
    public byte[] val;
    public eDoodleCategory label;
}

public class Category
{
    public DoodleData[] training;
    public DoodleData[] testing;
}

public enum eClassifierState
{
    NONE,
    TRAIN
}

public class Classifier : MonoBehaviour {
    public UI ui;
    public Painter painter;
    public eClassifierState classifierState = eClassifierState.NONE;

    private static string trainedModelFile = "/Resources/Data/nn.json";
    private static int len = 784;
    private static int total_data = 1000;
    private static float useTrainingData = 0.8f;
    private static int img_width = 28;
    private static int img_height = 28;
    private float[] inputImg = new float[len];

    private byte[] cats_data;
    private byte[] rainbows_data;
    private byte[] trains_data;

    private Category cats = new Category();
    private Category rainbows = new Category();
    private Category trains = new Category();

    private DoodleData[] training;
    private DoodleData[] testing;

    private Texture2D targetTex;
    private Texture2D nTex;

    private NeuralNetwork nn;

    // Use this for initialization
    void Start () {
        CreateData();
    }

    private void CreateData()
    {
        TextAsset catdata = Resources.Load("Data/cat1000.bin") as TextAsset;
        cats_data = catdata.bytes;

        TextAsset rainbowdata = Resources.Load("Data/rainbows1000.bin") as TextAsset;
        rainbows_data = rainbowdata.bytes;

        TextAsset traindata = Resources.Load("Data/train1000.bin") as TextAsset;
        trains_data = traindata.bytes;

        PrepareData(cats, cats_data, eDoodleCategory.CAT);
        PrepareData(rainbows, rainbows_data, eDoodleCategory.RAINBOW);
        PrepareData(trains, trains_data, eDoodleCategory.TRAIN);

        // input:784, hidden:64, output:3
        nn = new NeuralNetwork(784, 64, 3);

        training = new DoodleData[0];
        training = training.Concat(cats.training).ToArray();
        training = training.Concat(rainbows.training).ToArray();
        training = training.Concat(trains.training).ToArray();

        testing = new DoodleData[0];
        testing = testing.Concat(cats.testing).ToArray();
        testing = testing.Concat(rainbows.testing).ToArray();
        testing = testing.Concat(trains.testing).ToArray();
    }

    private void PrepareData(Category category, byte[] data, eDoodleCategory label)
    {
        int trainingdata_num = (int)(useTrainingData * total_data);
        category.training = new DoodleData[trainingdata_num];
        category.testing = new DoodleData[total_data-trainingdata_num];

        for (int i = 0; i< total_data; i++)
        {
            int offset = i * len;
            int strat = offset;
            int end = (offset + len)-1;
            if (i< trainingdata_num)
            {
                // training data
                category.training[i] = new DoodleData();
                category.training[i].val = NNUtils.SubArray(data, strat, end);
                category.training[i].label = label;
            }
            else
            {
                // testing data
                category.testing[i - trainingdata_num] = new DoodleData();
                category.testing[i - trainingdata_num].val = NNUtils.SubArray(data, strat, end);
                category.testing[i - trainingdata_num].label = label;
            }
        }
    }

    public void TrainEpoch()
    {
        classifierState = eClassifierState.TRAIN;
        training = NNUtils.Shuffle(training);
        for (int i = 0; i< training.Length; i++)
        {
            DoodleData data = training[i];
            List<float> inputs = new List<float>();
            for (int j = 0; j < data.val.Length; j++)
            {
                inputs.Add(data.val[j] / 255.0f);
            }

            eDoodleCategory label = data.label;
            List<float> targets = new List<float>() { 0f, 0f, 0f};
            targets[(int)label] = 1;

            nn.train(inputs, targets);
        }
        classifierState = eClassifierState.NONE;
    }

    public float TestAll()
    {
        int correct = 0;
        for (int i = 0; i < testing.Length; i++)
        {
            DoodleData data = testing[i];
            List<float> inputs = new List<float>();
            for (int j = 0; j < data.val.Length; j++)
            {
                inputs.Add(data.val[j] / 255.0f);
            }

            eDoodleCategory label = data.label;
            float[] guess = nn.predict(inputs).ToArray();
            float m = Mathf.Max(guess);
            int classification = System.Array.IndexOf(guess, m);
            if (classification == (int)label)
            {
                correct++;
            }
        }

        float percent = 100f * (float)correct / testing.Length;
        return percent;
    }

    public void ProcessImage()
    {
        targetTex = Resize(painter.GetTexture(), img_width, img_height);// resize 28 x 28 texture
        ui.rimg.texture = targetTex; // Make sure that the resize works ok.

        // make 1d array for inputs
        Color[] pix = targetTex.GetPixels(0, 0, img_width, img_height);
        List<float> inputs = new List<float>();
        for (int i = 0; i< pix.Length; i++)
        {
            Color col = pix[i];
            inputs.Add(col.r);
        }

        /*
        Texture2D destTex = new Texture2D(img_width, img_height);
        destTex.SetPixels(pix);
        destTex.Apply();
        ui.rimg.texture = destTex;
        */

        float[] guess = nn.predict(inputs).ToArray();
        float m = Mathf.Max(guess);
        int classification = System.Array.IndexOf(guess, m);

        if (classification == (int)eDoodleCategory.CAT)
        {
            ui.resText.text = "RESULT: "+eDoodleCategory.CAT.ToString();
        }
        else if (classification == (int)eDoodleCategory.RAINBOW)
        {
            ui.resText.text = "RESULT: " + eDoodleCategory.RAINBOW.ToString();
        }
        else if (classification == (int)eDoodleCategory.TRAIN)
        {
            ui.resText.text = "RESULT: " + eDoodleCategory.TRAIN.ToString();
        }
    }

    public void SaveModel()
    {
        string jsonstr = nn.ToJSON();
        NNUtils.SaveText(trainedModelFile, jsonstr);
        Debug.Log(jsonstr);
    }

    public void LoadModel()
    {
        string jsonTxt = NNUtils.LoadText(trainedModelFile);
        nn = NeuralNetwork.CreateFromJSON(jsonTxt);
    }

    private Texture2D Resize(Texture2D source, int newWidth, int newHeight)
    {
        source.filterMode = FilterMode.Point;
        RenderTexture rt = RenderTexture.GetTemporary(newWidth, newHeight);
        rt.filterMode = FilterMode.Point;
        RenderTexture.active = rt;
        Graphics.Blit(source, rt);
        nTex = new Texture2D(newWidth, newHeight);
        nTex.ReadPixels(new Rect(0, 0, newWidth, newWidth), 0, 0);

        nTex = FlipImage(nTex);

        nTex.Apply();
        RenderTexture.active = null;
        return nTex;
    }

    private Texture2D FlipImage(Texture2D source)
    {
        var srcPixels = source.GetPixels();
        var outPixels = new Color[srcPixels.Length];

        var currentIndex = 0;

        var startX = source.width - 1;
        var startY = source.height - 1;

        for (var y = startY; y >= 0; y--)
        {
            for (var x = startX; x >= 0; x--)
            {
                outPixels[currentIndex++] = srcPixels[y * source.width + x];
            }
        }

        source.SetPixels(outPixels);

        return source;
    }
}
