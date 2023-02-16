using System;

namespace WessleyMitchell.NeuralNetworking.Functions
{
    public class SigmoidFunction : IActivationFunction
    {
        public double CalculateOutput(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

        public double CalculateDerivative(double input)
        {
            double output = CalculateOutput(input);
            return output * (1 - output);
        }
    }
}
