namespace WessleyMitchell.NeuralNetworking.Functions
{
    public class IdentityFunction : IActivationFunction
    {
        public double CalculateOutput(double input)
        {
            return input;
        }

        public double CalculateDerivative(double input)
        {
            return 1;
        }
    }
}
