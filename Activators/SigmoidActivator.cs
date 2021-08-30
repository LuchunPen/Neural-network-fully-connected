using System;

namespace Neural_SL
{
    public class SigmoidActivator : NeuralActivator
    {
        public override double DF(double x)
        {
            double sigmoid = F(x);
            double result = sigmoid * (1 - sigmoid);

            return result;
        }

        public override double F(double x)
        {
            double result = 1.0 / (1.0 + Math.Exp(-x));
            return result;
        }
    }
}
