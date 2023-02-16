using WessleyMitchell.NeuralNetworking.Neurons;
using WessleyMitchell.NeuralNetworking.RandomNumberGenerators;

namespace WessleyMitchell.NeuralNetworking.Layers
{
    internal static class ILayerExtensions
    {
        /// <summary>
        /// Connects neurons of the <paramref name="inputOrHiddenLayer"/> with the neurons of the <paramref name="outputLayer"/>
        /// </summary>
        internal static void ConnectOutputLayer(this ILayer inputOrHiddenLayer, HiddenOrOutputLayer outputLayer, IRandomNumberGenerator randomNumberGenerator)
        {
            foreach (INeuron inputNeuron in inputOrHiddenLayer.Neurons)
            {
                foreach (HiddenOrOutputNeuron outputNeuron in outputLayer.Neurons)
                {
                    inputNeuron.ConnectNeurons(outputNeuron, randomNumberGenerator);
                }
            }
        }
    }
}
