using WessleyMitchell.NeuralNetworking.RandomNumberGenerators;

namespace WessleyMitchell.NeuralNetworking.Neurons
{
    internal static class INeuronExtensions
    {
        /// <summary>
        /// Connect two neurons. 
        /// This neuron is the output neuron of the connection.
        /// </summary>
        /// <param name="outputNeuron">Neuron that will be input neuron of the newly created connection.</param>
        internal static void ConnectNeurons(this INeuron inputNeuron, HiddenOrOutputNeuron outputNeuron, IRandomNumberGenerator randomNumberGenerator)
        {
            double weight = randomNumberGenerator.Next();
            var synapse = new Synapse(inputNeuron, outputNeuron, weight);
            inputNeuron.AddOutput(synapse);
            outputNeuron.AddInput(synapse);
        }
    }
}
