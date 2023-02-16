namespace WessleyMitchell.NeuralNetworking.Functions
{
    public interface IErrorFunction
    {
        double CalculateOutput(double expected, double actual);
        double CalculateDerivative(double expected, double actual);
    }
}
