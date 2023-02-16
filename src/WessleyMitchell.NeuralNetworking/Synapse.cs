using System;
using System.Collections.Generic;
using System.Linq;
using WessleyMitchell.NeuralNetworking.Neurons;

namespace WessleyMitchell.NeuralNetworking
{
    public class Synapse
    {
        /// <summary>
        /// Previous neuron to which this <see cref="Synapse"/> is connected
        /// </summary>
        private readonly INeuron inputNeuron;

        /// <summary>
        /// Next neuron to which this <see cref="Synapse"/> is connected
        /// </summary>
        private readonly HiddenOrOutputNeuron outputNeuron;

        /// <summary>
        /// Weight of the <see cref="Synapse"/>
        /// </summary>
        public double Weight { get; set; }

        /// <summary>
        /// Used to accumulate weights for future averaging
        /// </summary>
        private readonly List<double> weightDeltas;
        internal Synapse(INeuron inputNeuron, HiddenOrOutputNeuron outputNeuron, double weight)
        {
            this.inputNeuron = inputNeuron;
            this.outputNeuron = outputNeuron;
            Weight = weight;
            weightDeltas = new List<double>();
        }

        /// <summary>
        /// Get the output of the input neuron
        /// </summary>
        internal double GetOutput()
        {
            return inputNeuron.CalculateActivation();
        }

        /// <summary>
        /// Get the error of the output neuron
        /// </summary>
        internal double OutputNeuronError => outputNeuron.Error;

        /// <summary>
        /// Construct a list to track weight deltas.
        /// </summary>
        internal void ClearWeightDeltas()
        {
            weightDeltas.Clear();
        }

        /// <summary>
        /// Add to weight deltas for future averaging and adding to current weight.  (Some experts recommend weight averaging, and some do not.)
        /// </summary>
        internal void AddToWeightDeltas(double weightDelta)
        {
            weightDeltas.Add(weightDelta);
        }

        /// <summary>
        /// Sets <see cref="Weight"/> based on the average of collected weight deltas.  (Some experts recommend weight averaging, and some do not.)
        /// </summary>
        internal void CalculateNewWeight()
        {
            if (!weightDeltas.Any()) { throw new InvalidOperationException("No weights have been added."); }
            double weightAddend = weightDeltas.Sum() / weightDeltas.Count;
            Weight += weightAddend;
        }
    }
}
