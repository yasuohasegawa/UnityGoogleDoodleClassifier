using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// original code from Daniel Shiffman https://github.com/CodingTrain/Toy-Neural-Network-JS
// The Json serialize and deserialize are not considering about performance.
namespace NuralNet
{
    // d means the Derivatives of activation functions
    public class ActivateFunction
    {
        public static System.Func<float, float> sigmoid = (x) => {
            return 1.0f / (1.0f + Mathf.Exp(-x));
        };

        public static System.Func<float, float> dsigmoid = (y) => {
            return y * (1.0f - y);
        };

        public static System.Func<float, float> tanh = (x) => {
            return (float)System.Math.Tanh(x);
        };

        public static System.Func<float, float> dtanh = (y) => {
            return 1.0f-(y*y);
        };
    }

    [System.Serializable]
    public class NeuralNetwork
    {
        public int input_nodes;
        public int hidden_nodes;
        public int output_nodes;

        public NNMatrix weights_ih;
        public NNMatrix weights_ho;

        public NNMatrix bias_h;
        public NNMatrix bias_o;

        public float learning_rate = 0.1f;

        [System.NonSerialized]
        public System.Func<float, float> activate = ActivateFunction.sigmoid;

        [System.NonSerialized]
        public System.Func<float, float> dactivate = ActivateFunction.dsigmoid;

        public NeuralNetwork(int _input_nodes, int _hidden_nodes, int _output_nodes)
        {
            this.input_nodes = _input_nodes;
            this.hidden_nodes = _hidden_nodes;
            this.output_nodes = _output_nodes;

            // weight between input and hidden
            this.weights_ih = new NNMatrix(this.hidden_nodes, this.input_nodes);
            this.weights_ih.randomize();

            // weight between hidden and output
            this.weights_ho = new NNMatrix(this.output_nodes, this.hidden_nodes);
            this.weights_ho.randomize();

            this.bias_h = new NNMatrix(this.hidden_nodes, 1);
            this.bias_o = new NNMatrix(this.output_nodes, 1);
        }

        public NeuralNetwork copy()
        {
            int _input_nodes = this.input_nodes;
            int _hidden_nodes = this.hidden_nodes;
            int _output_nodes = this.output_nodes;

            NNMatrix wih = this.weights_ih;
            NNMatrix who = this.weights_ho;

            float lr = this.learning_rate;

            NNMatrix bh = this.bias_h;
            NNMatrix oh = this.bias_o;

            NeuralNetwork nn = new NeuralNetwork(_input_nodes, _hidden_nodes, _output_nodes);
            nn.weights_ih = wih;
            nn.weights_ho = who;
            nn.learning_rate = lr;
            nn.bias_h = bh;
            nn.bias_o = oh;
            return nn;
        }

        public void mutate()
        {
            this.weights_ih = NNMatrix.map(this.weights_ih, NNUtils.mutate);
            this.weights_ho = NNMatrix.map(this.weights_ho, NNUtils.mutate);
        }

        // feedforward
        public List<float> predict(List<float> input_list)
        {
            NNMatrix inputs = NNMatrix.fromList(input_list);

            NNMatrix hidden = NNMatrix.multiply(this.weights_ih, inputs);
            hidden.add(this.bias_h);

            // activate function
            hidden.map(activate);

            NNMatrix output = NNMatrix.multiply(this.weights_ho, hidden);
            output.add(this.bias_o);
            output.map(activate);

            return output.toList();
        }

        public void train(List<float> input_list, List<float> target_list)
        {
            NNMatrix inputs = NNMatrix.fromList(input_list);

            NNMatrix hidden = NNMatrix.multiply(this.weights_ih, inputs);
            hidden.add(this.bias_h);
            // activate function
            hidden.map(activate);

            NNMatrix outputs = NNMatrix.multiply(this.weights_ho, hidden);
            outputs.add(this.bias_o);
            outputs.map(activate);

            NNMatrix targets = NNMatrix.fromList(target_list);

            // calculate the error
            // ERROR = TARGETS(answer) - OUTPUTS(actual outputs)
            NNMatrix output_errors = NNMatrix.subtract(targets, outputs);


            //var gradients = outputs * (1-outputs);
            // calculate gradient
            NNMatrix gradients = NNMatrix.map(outputs, dactivate);
            gradients.multiply(output_errors);
            gradients.multiply(this.learning_rate);

            // calculate deltas
            NNMatrix hidden_T = NNMatrix.transpose(hidden);
            NNMatrix weights_ho_deltas = NNMatrix.multiply(gradients, hidden_T);

            // Adjust the weights by deltas
            this.weights_ho.add(weights_ho_deltas);

            // Adjust bias by its delta (which is just the gradients)
            this.bias_o.add(gradients);

            // calculate the hidden layer errors
            NNMatrix who_t = NNMatrix.transpose(this.weights_ho);
            NNMatrix hidden_errors = NNMatrix.multiply(who_t, output_errors);

            // calculate hidden gradients
            NNMatrix hidden_gradient = NNMatrix.map(hidden, dactivate);
            hidden_gradient.multiply(hidden_errors);
            hidden_gradient.multiply(this.learning_rate);

            // calculate input->hidden deltas
            NNMatrix input_T = NNMatrix.transpose(inputs);
            NNMatrix weight_ih_deltas = NNMatrix.multiply(hidden_gradient, input_T);

            this.weights_ih.add(weight_ih_deltas);

            // Adjust bias by its delta (which is just the hiddent gradients)
            this.bias_h.add(hidden_gradient);
        }

        public static NeuralNetwork CreateFromJSON(string json)
        {
            NeuralNetwork nn = null;
            try
            {
                nn = JsonUtility.FromJson<NeuralNetwork>(json);
            }
            catch (System.Exception e)
            {

            }
            return nn;
        }

        public string ToJSON()
        {
            return JsonUtility.ToJson(this);
        }
    }
}