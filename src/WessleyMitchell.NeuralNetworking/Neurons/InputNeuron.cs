using System;
using System.Collections.Generic;

namespace WessleyMitchell.NeuralNetworking.Neurons
{
    internal class InputNeuron : INeuron
    {
        /// <summary>
        /// Cached activation value
        /// </summary>
        private Nullable<double> activation;

        private readonly List<Synapse> outputs;
        /// <summary>
        /// Collection of <see cref="Synapse"/> from this <see cref="InputNeuron"/> to the next <see cref="HiddenOrOutputNeuron"/>'s neurons
        /// </summary>
        public IEnumerable<Synapse> Outputs => outputs;

        public InputNeuron()
        {
            activation = null;
            outputs = new List<Synapse>();
        }

        public void AddOutput(Synapse synapse)
        {
            outputs.Add(synapse);
        }

        /// <summary>
        /// Returns the input value
        /// </summary>
        public double CalculateActivation()
        {
            if (activation != null) { return activation.Value; }
            throw new InvalidOperationException(nameof(activation) + " was null.");
        }

        /// <summary>
        /// Sets the input of the network as this <see cref="InputNeuron"/>'s <see cref="activation"/>
        /// </summary>
        internal void SetActivation(double output)
        {
            activation = output;
        }
    }
}
