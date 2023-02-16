using System;

namespace WessleyMitchell.NeuralNetworking.Functions
{
    public class MeanSquaredErrorFunction : IErrorFunction
    {
        public double CalculateOutput(double expected, double actual)
        {
            return .5 * Math.Pow(expected - actual, 2);
        }

        public double CalculateDerivative(double expected, double actual)
        {
            return expected - actual;
        }
    }
}
