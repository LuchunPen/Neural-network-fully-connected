using System;

namespace Neural_SL
{
    public class InputFullyConnectedNeuralLayer : FullyConnectedNeuralLayer
    {
        public InputFullyConnectedNeuralLayer(int neurons) 
            : base(neurons, 1, null, false)
        {
            for (int i = 0; i < neurons; i++)
            {
                _neuronsWeights[i, 0] = 1;
            }
        }

        public override double[,] Forward(double[,] inputs)
        {
            if (inputs == null) { throw new ArgumentNullException("Inputs data is null"); }
            int c_count = inputs.GetLength(0); int v_count = inputs.GetLength(1);
            if (c_count != NeuronsCount) { throw new ArgumentOutOfRangeException($"Wrong input data {c_count} to {NeuronsCount} neurons"); }
            if (v_count != 1) { throw new ArgumentOutOfRangeException($"Wrong input data {v_count}, must be 1"); }

            _neuronInputs = inputs;
            _neuronsOutputs = inputs;
            return _neuronsOutputs;
        }
    }
}
