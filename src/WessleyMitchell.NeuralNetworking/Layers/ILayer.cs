using System.Collections.Generic;
using WessleyMitchell.NeuralNetworking.Neurons;

namespace WessleyMitchell.NeuralNetworking.Layers
{
    public interface ILayer
    {
        IReadOnlyCollection<INeuron> Neurons { get; }
    }
}
