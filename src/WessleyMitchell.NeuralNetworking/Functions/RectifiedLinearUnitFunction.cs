using System;

namespace WessleyMitchell.NeuralNetworking.Functions
{
    public class RectifiedLinearUnitFunction : IActivationFunction
    {
        public double CalculateOutput(double input)
        {
            return Math.Max(0, input);
        }

        public double CalculateDerivative(double input)
        {
            if (input < 0) { return 0; }
            if (input > 0) { return 1; }
            throw new InvalidOperationException("undefined at " + input);
        }
    }
}
