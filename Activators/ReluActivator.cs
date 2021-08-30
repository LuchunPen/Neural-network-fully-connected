using System;

namespace Neural_SL
{
    public class ReluActivator : NeuralActivator
    {
        public override double DF(double x)
        {
            if (x < 0) { return 0; }
            else return 1;
        }

        public override double F(double x)
        {
            if (x < 0) { return 0; }
            return x;
        }
    }

    public class ReluLeakActivator : NeuralActivator
    {
        public override double DF(double x)
        {
            if (x < 0) { return 0.01; }
            else return 1;
        }

        public override double F(double x)
        {
            if (x < 0) { return 0.01 * x; }
            return x;
        }
    }
}
