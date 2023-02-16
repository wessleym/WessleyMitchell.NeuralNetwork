using System.Collections.Generic;

namespace WessleyMitchell.NeuralNetworking.Collections
{
    public class InputsAndOutputsItem
    {
        internal IReadOnlyCollection<double> Inputs { get; }
        internal IReadOnlyCollection<double> Outputs { get; }
        internal InputsAndOutputsItem(IReadOnlyCollection<double> inputs, IReadOnlyCollection<double> outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }
    }
}
