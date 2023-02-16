using System.Collections.Generic;

namespace WessleyMitchell.NeuralNetworking.Functions
{
    public interface IInputFunction
    {
        double CalculateInput(IEnumerable<Synapse> inputs);
    }
}
