using System.Collections.Generic;
using WessleyMitchell.NeuralNetworking.Neurons;

namespace WessleyMitchell.NeuralNetworking.Layers
{
    /// <summary>
    /// A layer with <see cref="HiddenOrOutputNeuron"/> neurons
    /// </summary>
    public class HiddenOrOutputLayer : LayerBase, ILayer
    {
        public IReadOnlyCollection<HiddenOrOutputNeuron> Neurons { get; }
        IReadOnlyCollection<INeuron> ILayer.Neurons => Neurons;
        internal HiddenOrOutputLayer(IReadOnlyCollection<HiddenOrOutputNeuron> neurons)
            : base(neurons)
        {
            Neurons = neurons;
        }
    }
}
