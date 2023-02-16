namespace WessleyMitchell.NeuralNetworking.Functions
{
    public interface IActivationFunction
    {
        double CalculateOutput(double input);
        double CalculateDerivative(double input);
    }
}
