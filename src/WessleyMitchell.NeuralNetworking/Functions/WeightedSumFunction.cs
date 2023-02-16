using System.Collections.Generic;
using System.Linq;

namespace WessleyMitchell.NeuralNetworking.Functions
{
    public class WeightedSumFunction : IInputFunction
    {
        public double CalculateInput(IEnumerable<Synapse> inputs)
        {
            return inputs.Select(x => x.Weight * x.GetOutput()).Sum();
        }
    }
}
