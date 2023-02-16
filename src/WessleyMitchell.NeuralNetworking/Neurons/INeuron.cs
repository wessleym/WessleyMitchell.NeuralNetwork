using System.Collections.Generic;

namespace WessleyMitchell.NeuralNetworking.Neurons
{
    /// <summary>
    /// Defines <see cref="Synapse"/> outputs and calculated output value
    /// </summary>
    public interface INeuron
    {
        /// <summary>
        /// Output <see cref="Synapse"/>s leading to the next layer
        /// </summary>
        IEnumerable<Synapse> Outputs { get; }

        /// <summary>
        /// Add <see cref="Synapse"/> to <see cref="Outputs"/>
        /// </summary>
        void AddOutput(Synapse output);

        /// <summary>
        /// For <see cref="InputNeuron"/>s, returns the input value
        /// For <see cref="HiddenOrOutputNeuron"/>s, calculates the activation after summing inputs and applying the activation function (and possibly applying a bias).
        /// </summary>
        double CalculateActivation();
    }
}
