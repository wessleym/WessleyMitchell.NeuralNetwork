using System.Collections.Generic;
using System.Linq;
using System.Threading;
using WessleyMitchell.NeuralNetworking.Functions;
using WessleyMitchell.NeuralNetworking.Neurons;

namespace WessleyMitchell.NeuralNetworking.Layers
{
    /// <summary>
    /// Factory used to create layers.
    /// </summary>
    public static class LayerFactory
    {
        /// <summary>
        /// Returns as many <see cref="InputNeuron"/>s as specified by <paramref name="neuronCount"/>
        /// </summary>
        private static IEnumerable<InputNeuron> ConstructInputNeurons(int neuronCount)
        {
            for (int i = 0; i < neuronCount; i++)
            {
                InputNeuron neuron = new InputNeuron();
                yield return neuron;
            }
        }

        /// <summary>
        /// Returns as many <see cref="HiddenOrOutputNeuron"/>s as specified by <paramref name="neuronCount"/>
        /// </summary>
        private static IEnumerable<HiddenOrOutputNeuron> ConstructHiddenOrOutputNeurons(int neuronCount, IActivationFunction activationFunction, IInputFunction inputFunction)
        {
            for (int i = 0; i < neuronCount; i++)
            {
                HiddenOrOutputNeuron neuron = new HiddenOrOutputNeuron(activationFunction, inputFunction);
                yield return neuron;
            }
        }

        /// <summary>
        /// Constructs an <see cref="InputLayer"/> with as many <see cref="InputNeuron"/>s as specified by <paramref name="neuronCount"/>
        /// </summary>
        internal static InputLayer ConstructInputLayer(int neuronCount)
        {
            InputNeuron[] neurons = ConstructInputNeurons(neuronCount).ToArray();
            return new InputLayer(neurons);
        }


        /// <summary>
        /// Constructs a <see cref="HiddenOrOutputLayer"/> with as many <see cref="HiddenOrOutputNeuron"/>s as specified by <paramref name="neuronCount"/>
        /// </summary>
        public static HiddenOrOutputLayer ConstructHiddenOrOutputLayer(int neuronCount, IActivationFunction activationFunction, IInputFunction inputFunction)
        {//For all neural networks, neurons in the same layer will almost always have the same activation function.
            //But since they don't always, I am allowing for future indepedent neuron activation function configuration.
            HiddenOrOutputNeuron[] neurons = ConstructHiddenOrOutputNeurons(neuronCount, activationFunction, inputFunction).ToArray();
            return new HiddenOrOutputLayer(neurons);
        }
    }
}
