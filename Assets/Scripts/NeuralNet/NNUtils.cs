using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

namespace NuralNet
{
    public class NNUtils
    {
        // original code from Daniel Shiffman https://github.com/CodingTrain/Toy-Neural-Network-JS
        public static System.Func<float, float> mutate = (x) =>
        {
            if (UnityEngine.Random.Range(0f, 1.0f) < 0.1f)
            {
                float offset = UnityEngine.Random.Range(-0.1f, 0.1f);
                float newx = x + offset;
                return newx;
            }
            return x;
        };

        public static bool SaveText(string path, string text)
        {
            try
            {
                using (StreamWriter writer = new StreamWriter(Application.dataPath + path, false))
                {
                    writer.Write(text);
                    writer.Flush();
                    writer.Close();
                }
            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
                return false;
            }
            return true;
        }

        public static string LoadText(string path)
        {
            string strStream = "";
            try
            {
                using (StreamReader sr = new StreamReader(Application.dataPath + path))
                {
                    strStream = sr.ReadToEnd();
                    sr.Close();
                }
            }
            catch (Exception e)
            {
                Debug.Log(e.Message);
            }

            return strStream;
        }

        public static T[] SubArray<T>(T[] source, int start, int end)
        {
            int count = end - start + 1;
            T[] result = new T[count];
            Array.Copy(source, start, result, 0, count);

            return result;
        }

        //http://www.codeproject.com/Articles/35114/Shuffling-arrays-in-C
        // Change randomly the order of the array.
        public static T[] Shuffle<T>(T[] array) { return Shuffle<T>(array, 0, array.Length - 1); }
        // Change randomly the order of a part of the array.
        public static T[] Shuffle<T>(T[] array, int start, int end)
        {
            int count = end - start + 1;
            T[] shuffledPart = new T[count];
            Array.Copy(array, start, shuffledPart, 0, count);

            var matrix = new SortedList();
            var r = new System.Random();

            for (var x = 0; x <= shuffledPart.GetUpperBound(0); x++)
            {
                var i = r.Next();
                while (matrix.ContainsKey(i)) { i = r.Next(); }
                matrix.Add(i, shuffledPart[x]);
            }

            matrix.Values.CopyTo(shuffledPart, 0);
            T[] result = (T[])array.Clone();
            Array.Copy(shuffledPart, 0, result, start, count);

            return result;
        }
    }
}