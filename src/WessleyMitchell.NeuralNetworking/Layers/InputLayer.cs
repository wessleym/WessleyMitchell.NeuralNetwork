using System.Collections.Generic;
using WessleyMitchell.NeuralNetworking.Neurons;

namespace WessleyMitchell.NeuralNetworking.Layers
{
    /// <summary>
    /// A layer with <see cref="InputNeuron"/> neurons
    /// </summary>
    internal class InputLayer : LayerBase, ILayer
    {
        public IReadOnlyCollection<InputNeuron> Neurons { get; }
        IReadOnlyCollection<INeuron> ILayer.Neurons => Neurons;
        public InputLayer(IReadOnlyCollection<InputNeuron> neurons)
            : base(neurons)
        {
            Neurons = neurons;
        }
    }
}
