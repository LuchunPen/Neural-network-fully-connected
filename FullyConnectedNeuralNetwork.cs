using System;

namespace Neural_SL
{
    public class FullyConnectedNeuralNetwork
    {
        private FullyConnectedNeuralLayer[] _layers;
        public FullyConnectedNeuralLayer[] Layers { get { return _layers; } }

        public FullyConnectedNeuralNetwork(InputFullyConnectedNeuralLayer inputLayer, params FullyConnectedNeuralLayer[] layers)
        {
            _layers = new FullyConnectedNeuralLayer[layers.Length + 1];
            _layers[0] = inputLayer;
            for (int i = 1; i < layers.Length; i++)
            {
                _layers[i] = layers[i - 1];
            }
        }

        public double[,] Calculate(double[,] inputs)
        {
            int inputs_data_count = inputs.GetLength(0);
            int inputs_neurons = _layers[0].NeuronsCount;

            if (inputs_data_count != inputs_neurons)
            { throw new ArgumentException($"Inputs data is incorrect: input data {inputs_data_count} / input neurons {inputs_neurons}"); }

            double[,] layer_input = inputs;
            for (int i = 0; i < _layers.Length; i++)
            {
                layer_input = _layers[i].Forward(layer_input);
            }
            return _layers[_layers.Length - 1].LastNeuronsOutputs;
        }

        /// <summary>
        /// Return result error
        /// </summary>
        /// <param name="data">Tuple of inputs and expected arrays</param>
        /// <param name="learningRate"></param>
        /// <returns></returns>
        public double Train(Tuple<double[,], double[,]> data, double learningRate)
        {
            double[,] inputs = data.Item1;
            double[,] expected = data.Item2;

            int inputs_data_count = inputs.GetLength(0);
            int inputs_neurons = _layers[0].NeuronsCount;
            int output_data_count = expected.GetLength(0);
            int output_neurons = _layers[_layers.Length - 1].NeuronsCount;

            if (inputs_data_count != inputs_neurons)
            { throw new ArgumentException($"Inputs data is incorrect: input data {inputs_data_count} / input neurons {inputs_neurons}"); }
            if (output_data_count != output_neurons)
            { throw new ArgumentException($"Expected data is incorrect expected data {output_data_count} / output neurons {output_neurons}"); }

            double[,] output = Calculate(inputs);
            double[,] error = NetworkError(output, expected);

            TrainInternal(error, learningRate);

            double result_error = ResultError(error);
            return result_error;
        }


        private void TrainInternal(double[,] error, double learningRate)
        {
            if (error == null || error.GetLength(0) != _layers[_layers.Length - 1].NeuronsCount)
            { throw new ArgumentException($"Wrong error table"); }

            double[,] l_error = error;

            //going throw layers from last to first
            for (int i = _layers.Length - 1; i >= 1; i--)
            {
                FullyConnectedNeuralLayer layer = _layers[i];
                double[,] layers_grad = layer.CorrectWeights(l_error, learningRate);

                // calculate error for each neurons in previous layer
                l_error = new double[layer.NeuronConnectionsCount, 1];

                for (int c = 0; c < layer.NeuronConnectionsCount; c++)
                {
                    double n_error = 0;
                    for (int n = 0; n < layer.NeuronsCount; n++)
                    {
                        n_error += layer.NeuronsWeights[n, c] * layers_grad[n, 0];
                    }
                    l_error[c, 0] = n_error;
                }
            }
        }

        private double[,] NetworkError(double[,] output, double[,] expected)
        {
            if (output == null || expected == null)
            { throw new ArgumentNullException("nullable table(s)"); }
            if (output.GetLength(0) != expected.GetLength(0))
            { throw new ArgumentException("output and target tables has different sizes"); }

            int count = output.GetLength(0);
            double[,] error = new double[count, 1];

            for (int i = 0; i < count; i++)
            {
                error[i, 0] = expected[i, 0] - output[i, 0];
            }

            return error;
        }

        private double ResultError(double[,] outputError)
        {
            if (outputError == null || outputError.GetLength(1) != 1)
            { throw new ArgumentNullException("wrong error table"); }

            double result = 0;

            for (int i = 0; i < outputError.GetLength(0); i++)
            {
                result += Math.Pow(outputError[i,0], 2);
            }

            return result / 2;
        }

        /*
        private double[,] Network_AbsError(double[,] output, double[,] expected)
        {
            if (output == null || expected == null)
            { throw new ArgumentNullException("nullable table(s)"); }
            if (output.GetLength(0) != expected.GetLength(0))
            { throw new ArgumentException("output and target tables has different sizes"); }

            int count = output.GetLength(0);
            double[,] error = new double[count, 1];

            for (int i = 0; i < count; i++)
            {
                error[i, 0] = Math.Abs(expected[i, 0] - output[i, 0]);
            }

            return error;
        }

        private double[,] Network_MeanSquareError(double[,] output, double[,] expected)
        {
            if (output == null || expected == null)
            { throw new ArgumentNullException("nullable table(s)"); }
            if (output.GetLength(0) != expected.GetLength(0))
            { throw new ArgumentException("output and target tables has different sizes"); }

            double sq_error = 0;
            int count = output.GetLength(0);
            for (int i = 0; i < count; i++)
            {
                double error = expected[i, 0] - output[i, 0];
                sq_error += (error * error);
            }
            sq_error /= count;
            double[,] result = new double[count, 1];
            for (int i = 0; i < count; i++)
            {
                result[i, 0] = sq_error;
            }
            return result;
        }

        private double[,] Network_CrossEntropyError(double[,] output, double[,] expected)
        {
            if (output == null || expected == null)
            { throw new ArgumentNullException("nullable table(s)"); }
            if (output.GetLength(0) != expected.GetLength(0))
            { throw new ArgumentException("output and target tables has different sizes"); }

            double sq_error = 0;
            int count = output.GetLength(0);
            for (int i = 0; i < count; i++)
            {
                double error = Math.Log(expected[i, 0]) * output[i, 0];
                sq_error += error;
            }
            sq_error = -1.0 * sq_error / count;
            double[,] result = new double[count, 1];
            for (int i = 0; i < count; i++)
            {
                result[i, 0] = sq_error;
            }

            return result;
        }
        */
    }
}
