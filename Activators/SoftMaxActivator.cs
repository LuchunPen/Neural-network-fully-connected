using System;

namespace Neural_SL
{
    public class SoftMaxActivator : NeuralActivator
    {
        public override double F(double x)
        {
            return 1;
        }

        public override double DF(double x)
        {
            return 0;
        }

        public override double[,] F(double[,] ax)
        {
            double[,] output = new double[ax.GetLength(0), 1];

            double summ = 0;
            for (int i = 0; i < ax.GetLength(0); i++)
            {
                output[i, 0] = Math.Exp(ax[i, 0]);
                summ += output[i, 0];
            }

            if (summ == 0) { return output; }

            for (int i = 0; i < output.Length; i++)
            {
                output[i, 0] = output[i, 0] / summ;
            }

            return output;
        }

        public override double[,] DF(double[,] ax)
        {
            double[,] output = F(ax);
            for (int i = 0; i < output.GetLength(0); i++)
            {
                double o = output[i, 0];
                output[i, 0] = o * (1 - o);
            }

            return output;
        }
    }
}
