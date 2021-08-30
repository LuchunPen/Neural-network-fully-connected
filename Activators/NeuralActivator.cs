using System;

namespace Neural_SL
{
    public abstract class NeuralActivator : INeuralActivator
    {
        public abstract double F(double x);

        public abstract double DF(double x);

        public virtual double[,] DF(double[,] ax)
        {
            double[,] output = new double[ax.GetLength(0), 1];
            for (int i = 0; i < ax.GetLength(0); i++)
            {
                output[i, 0] = DF(ax[i, 0]);
            }

            return output;
        }

        public virtual double[,] F(double[,] ax)
        {
            double[,] output = new double[ax.GetLength(0), 1];
            for (int i = 0; i < ax.GetLength(0); i++)
            {
                output[i, 0] = F(ax[i, 0]);
            }

            return output;
        }
    }
}
