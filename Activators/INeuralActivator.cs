namespace Neural_SL
{
    public interface INeuralActivator
    {
        double F(double x);
        double DF(double x);
        double[,] F(double[,] ax);
        double[,] DF(double[,] ax);
    }
}
