using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System.Linq;

// original code from Daniel Shiffman https://github.com/CodingTrain/Toy-Neural-Network-JS
// The Json serialize and deserialize are not considering about performance.
namespace NuralNet
{
    [System.Serializable]
    public class NNMatrix : ISerializationCallbackReceiver
    {
        public int rows;
        public int cols;

        [System.NonSerialized]
        public List<List<float>> data;

        public string[] data2d;

        public NNMatrix(int _rows, int _cols)
        {
            this.rows = _rows;
            this.cols = _cols;
            this.data = new List<List<float>>();

            for (int i = 0; i < rows; i++)
            {
                this.data.Add(new List<float>());
                for (var j = 0; j < this.cols; j++)
                {
                    this.data[i].Add(0f);
                }
            }
        }

        public void OnBeforeSerialize()
        {
            if (data != null)
            {
                data2d = new string[data.Count];
                for (int i = 0; i < data.Count; i++)
                {
                    data2d[i] = Join(",", data[i].ToArray());
                }
            }
        }

        public void OnAfterDeserialize()
        {
            this.data = new List<List<float>>();
            for (int i = 0; i < data2d.Length; i++)
            {
                string[] strArr = data2d[i].Split(',');
                float[] newData = System.Array.ConvertAll(strArr, s => float.Parse(s));
                data.Add(newData.ToList());
            }
        }

        private string Join<T>(string separator, T[] array)
        {
            StringBuilder builder = new StringBuilder();
            foreach (T value in array)
            {
                builder.Append(value);
                builder.Append(separator);
            }

            // remove last character
            string s = builder.ToString();
            s = s.Remove(s.Length - 1);
            return s;
        }

        public static NNMatrix fromList(List<float> list)
        {
            NNMatrix m = new NNMatrix(list.Count, 1);
            for (int i = 0; i < list.Count; i++)
            {
                m.data[i][0] = list[i];
            }
            return m;
        }

        public static NNMatrix subtract(NNMatrix a, NNMatrix b)
        {
            NNMatrix result = new NNMatrix(a.rows, a.cols);
            for (int i = 0; i < result.rows; i++)
            {
                for (int j = 0; j < result.cols; j++)
                {
                    result.data[i][j] = a.data[i][j] - b.data[i][j];
                }
            }
            return result;
        }

        public List<float> toList()
        {
            List<float> list = new List<float>();
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    list.Add(this.data[i][j]);
                }
            }
            return list;
        }

        public void randomize()
        {
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    this.data[i][j] = Random.Range(-1f, 1f);
                }
            }
        }

        public void add<Type>(Type n)
        {
            if (n is NNMatrix)
            {
                for (int i = 0; i < this.rows; i++)
                {
                    for (int j = 0; j < this.cols; j++)
                    {
                        NNMatrix mat = (NNMatrix)(object)n;
                        this.data[i][j] += mat.data[i][j];
                    }
                }
            }
            else
            {
                for (int i = 0; i < this.rows; i++)
                {
                    for (int j = 0; j < this.cols; j++)
                    {
                        float val = (float)(object)n;
                        this.data[i][j] += val;
                    }
                }
            }
        }

        public static NNMatrix transpose(NNMatrix matrix)
        {
            NNMatrix result = new NNMatrix(matrix.cols, matrix.rows);

            for (int i = 0; i < matrix.rows; i++)
            {
                for (int j = 0; j < matrix.cols; j++)
                {
                    result.data[j][i] = matrix.data[i][j];
                }
            }
            return result;
        }

        public static NNMatrix multiply(NNMatrix a, NNMatrix b)
        {
            if (a.cols != b.rows)
            {
                Debug.Log("Columns of A must match rows of B.");
                return null;
            }

            NNMatrix result = new NNMatrix(a.rows, b.cols);

            for (int i = 0; i < result.rows; i++)
            {
                for (int j = 0; j < result.cols; j++)
                {

                    float sum = 0f;
                    for (int k = 0; k < a.cols; k++)
                    {
                        sum += a.data[i][k] * b.data[k][j];
                    }

                    result.data[i][j] = sum;
                }
            }

            return result;
        }


        public void multiply<Type>(Type n)
        {
            if (n is NNMatrix)
            {
                for (int i = 0; i < this.rows; i++)
                {
                    for (int j = 0; j < this.cols; j++)
                    {
                        NNMatrix mat = (NNMatrix)(object)n;
                        this.data[i][j] *= mat.data[i][j];
                    }
                }
            }
            else
            {
                for (int i = 0; i < this.rows; i++)
                {
                    for (int j = 0; j < this.cols; j++)
                    {
                        float val = (float)(object)n;
                        this.data[i][j] *= val;
                    }
                }
            }
        }

        public void map(System.Func<float, float> fn)
        {
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    float val = this.data[i][j];
                    this.data[i][j] = fn(val);
                }
            }
        }

        public static NNMatrix map(NNMatrix matrix, System.Func<float, float> func)
        {
            NNMatrix result = new NNMatrix(matrix.rows, matrix.cols);
            for (int i = 0; i < matrix.rows; i++)
            {
                for (int j = 0; j < matrix.cols; j++)
                {
                    float val = matrix.data[i][j];
                    result.data[i][j] = func(val);
                }
            }

            return result;
        }

    }
}