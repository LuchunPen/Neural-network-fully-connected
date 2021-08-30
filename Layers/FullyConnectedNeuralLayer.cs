using System;

namespace Neural_SL
{
    public class FullyConnectedNeuralLayer
    {
        protected double[,] _neuronsWeights;
        public double[,] NeuronsWeights { get { return _neuronsWeights; } }
        public int NeuronsCount { get { return NeuronsWeights.GetLength(0); } }
        public int NeuronConnectionsCount { get { return NeuronsWeights.GetLength(1); } }

        protected double[,] _biasWeights;
        public double[,] BiasWeights { get { return _biasWeights; } }

        protected INeuralActivator _activator;
        public INeuralActivator Activator { get { return _activator; } }

        protected double[,] _neuronInputs;
        public double[,] LastNeuronsInputs { get { return _neuronInputs; } }

        protected double[,] _neuronsOutputs;
        public double[,] LastNeuronsOutputs { get { return _neuronsOutputs; } }

        protected double[,] _localGradient;
        public double[,] LastLocalGradient { get { return _localGradient; } }

        protected double _learningRateRatio = 1;
        public double LearningRateRatio
        {
            get { return _learningRateRatio; }
            set { if (value > 0) { _learningRateRatio = value; } }
        }


        /// <param name="neurons">min 1</param>
        /// <param name="connections">min 1</param>
        /// <param name="activator">activation func and func derivative </param>
        /// <param name="setRandomWeights">autofill weights for each neuron by values from 0 to 1</param>
        public FullyConnectedNeuralLayer(int neurons, int connections, INeuralActivator activator, bool setRandomWeights = true)
        {
            if (neurons < 1) { throw new ArgumentException("Neurons count must be more than 0"); }
            if (connections < 1) { throw new ArgumentException("Neurons must have connections"); }

            _neuronsWeights = new double[neurons, connections];
            _neuronsOutputs = new double[neurons, 1];
            _neuronInputs = new double[connections, 1];
            _biasWeights = new double[neurons, 1];
            _activator = activator;

            if (setRandomWeights) { SetRandomWeights(); }
        }

        /// <param name="weights">neurons weights matrix: vertical - neurons, horizontal - neurons weights, </param>
        /// <param name="activator">neurons layer acivator</param>
        public FullyConnectedNeuralLayer(double[,] weights, INeuralActivator activator)
        {
            if (weights == null || weights.GetLength(0) < 1 || weights.GetLength(1) < 1) {
                throw new ArgumentException("Wrong weights table");
            }
            _neuronsWeights = weights;
            _activator = activator;

            _biasWeights = new double[NeuronsCount, 1];
            _neuronsOutputs = new double[NeuronsCount, 1];
            _neuronInputs = new double[NeuronConnectionsCount, 1];
        }

        /// <param name="weights">neurons weights matrix: vertical - neurons, horizontal - neurons weights, </param>
        /// <param name="biases">each neurons bias weights</param>
        /// <param name="activator">neurons layer acivator</param>
        public FullyConnectedNeuralLayer(double[,] weights, double[] biases, INeuralActivator activator)
        {
            if (weights == null || weights.GetLength(0) < 1 || weights.GetLength(1) < 1) {
                throw new ArgumentException("Wrong weights table");
            }

            if (biases == null || biases.Length != weights.GetLength(0)) {
                throw new ArgumentException("Wrong biases table count");
            }

            _neuronsWeights = weights;
            _biasWeights = new double[biases.Length, 1];
            for (int i = 0; i < biases.Length; i++)
            {
                _biasWeights[i, 0] = biases[i];
            }

            _activator = activator;

            _neuronsOutputs = new double[NeuronsCount, 1];
            _neuronInputs = new double[NeuronConnectionsCount, 1];
        }

        protected void SetRandomWeights()
        {
            Random r = new Random();
            for (int n = 0; n < _neuronsWeights.GetLength(0); n++)
            {
                for (int c = 0; c < _neuronsWeights.GetLength(1); c++)
                {
                    _neuronsWeights[n, c] = r.NextDouble() - 0.5;
                }

                _biasWeights[n, 0] = r.NextDouble() - 0.5;
            }
        }

        public double[,] GetNeuronWeights(int neuron)
        {
            if (neuron >= _neuronsWeights.GetLength(0)) { throw new ArgumentOutOfRangeException(); }

            double[,] result = new double[1, _neuronsWeights.GetLength(1)];
            for (int i = 0; i < result.Length; i++)
            {
                result[0, i] = _neuronsWeights[neuron, i];
            }

            return result;
        }

        public virtual double[,] Forward(double[,] inputs)
        {
            double[,] signals = CalculateSignal(inputs);

            signals = _activator.F(signals);
            _neuronInputs = inputs;
            _neuronsOutputs = signals;

            return _neuronsOutputs;
        }

        protected virtual double[,] CalculateSignal(double[,] inputs)
        {
            if (inputs == null) { throw new ArgumentNullException("inpts data is null"); }
            int c_count = inputs.GetLength(0); int v_count = inputs.GetLength(1);
            if (c_count != NeuronConnectionsCount) { throw new ArgumentOutOfRangeException($"Wrong input data {c_count} to {NeuronConnectionsCount} connections"); }
            if (v_count != 1) { throw new ArgumentOutOfRangeException($"Wrong input data {v_count}, must be 1"); }

            double[,] result = new double[NeuronsCount, 1];
            for (int n = 0; n < NeuronsCount; n++)
            {
                double n_output = 0;
                for (int c = 0; c < c_count; c++)
                {
                    n_output += _neuronsWeights[n, c] * inputs[c, 0];
                }
                result[n, 0] = n_output + _biasWeights[n, 0];
            }

            return result;
        }

        public virtual double[,] CorrectWeights(double[,] error, double learning_rate)
        {
            if (error.GetLength(0) != NeuronsCount || error.GetLength(1) != 1) { throw new ArgumentException("Wrong error data"); }
            
            _localGradient = new double[NeuronsCount, 1];
            double[,] signals = CalculateSignal(LastNeuronsInputs);
            double[,] d_signals = _activator.DF(signals);

            // weights delta for each neurons
            for (int n = 0; n < NeuronsCount; n++)
            {
                double grad = error[n, 0] * d_signals[n, 0];
                _localGradient[n, 0] = grad;
                //for each neuron correcting weights
                for (int c = 0; c < NeuronConnectionsCount; c++)
                {
                    double delta_w = _neuronInputs[c, 0] * grad * learning_rate;
                    _neuronsWeights[n, c] += delta_w;
                }

                _biasWeights[n, 0] += grad * learning_rate;
            }

            return _localGradient;
        }

        public override string ToString()
        {
            return $"{NeuronsCount} neurons with {NeuronConnectionsCount} connections";
        }

    }
}
