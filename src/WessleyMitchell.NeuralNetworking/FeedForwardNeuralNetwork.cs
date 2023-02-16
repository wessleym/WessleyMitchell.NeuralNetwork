using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using WessleyMitchell.NeuralNetworking.Layers;
using WessleyMitchell.NeuralNetworking.Neurons;
using WessleyMitchell.NeuralNetworking.RandomNumberGenerators;

namespace WessleyMitchell.NeuralNetworking
{
    public class FeedForwardNeuralNetwork : IEnumerable<ILayer>
    {
        /// <summary>
        /// The input layer, which is constructed initially with the neural network
        /// </summary>
        private readonly InputLayer inputLayer;

        /// <summary>
        /// The hidden and output layers, which should be added by the caller immediately after construction of the neural network
        /// </summary>
        public List<HiddenOrOutputLayer> HiddenAndOutputLayers { get; }

        /// <summary>
        /// Specifies how random numbers will be generated for the network
        /// </summary>
        private readonly IRandomNumberGenerator randomNumberGenerator;

        /// <summary>
        /// Set up input layers and a collection of hidden and output layers
        /// </summary>
        /// <param name="inputNeuronCount">Input neuron count</param>
        /// <param name="randomNumberGenerator">Specifies how random numbers will be generated for the network</param>
        public FeedForwardNeuralNetwork(int inputNeuronCount, IRandomNumberGenerator randomNumberGenerator)
        {
            inputLayer = LayerFactory.ConstructInputLayer(inputNeuronCount);
            HiddenAndOutputLayers = new List<HiddenOrOutputLayer>();
            this.randomNumberGenerator = randomNumberGenerator;
        }

        public IEnumerator<ILayer> GetEnumerator()
        {
            yield return inputLayer;
            foreach (HiddenOrOutputLayer layer in HiddenAndOutputLayers)
            {
                yield return layer;
            }
        }
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <summary>
        /// Add new last layer.
        /// </summary>
        public void Add(HiddenOrOutputLayer newLayer)
        {
            ILayer lastLayer = this.Last();
            //Connect new layer as output of last layer.
            lastLayer.ConnectOutputLayer(newLayer, randomNumberGenerator);
            HiddenAndOutputLayers.Add(newLayer);
        }

        /// <summary>
        /// Returns all hidden layers (i.e., all layers from <see cref="HiddenAndOutputLayers"/> except the last one, which is the <see cref="OutputLayer"/>)
        /// </summary>
        internal IEnumerable<HiddenOrOutputLayer> HiddenLayers => HiddenAndOutputLayers.Take(HiddenAndOutputLayers.Count - 1);

        /// <summary>
        /// Returns the last of the <see cref="HiddenAndOutputLayers"/>
        /// </summary>
        internal HiddenOrOutputLayer OutputLayer => HiddenAndOutputLayers.Last();

        /// <summary>
        /// Set input values, which are stored in the first layer
        /// </summary>
        private void SetInputLayerOutputsAndResetOtherOutputs(IReadOnlyCollection<double> inputs)
        {
            IReadOnlyCollection<InputNeuron> inputLayerNeurons = inputLayer.Neurons;
            if (inputLayerNeurons.Count != inputs.Count)
            {
                throw new InvalidOperationException("The number of first-layer neurons (" + inputLayerNeurons.Count + ") did not equal the number of inputs (" + inputs.Count + ").");
            }
            foreach (var inputAndNeuron in inputLayerNeurons.Zip(inputs, (neuron, input) => new { neuron, input }))
            {
                //For each input, set input value to activation value
                inputAndNeuron.neuron.SetActivation(inputAndNeuron.input);
            }
            foreach (ILayer layer in HiddenAndOutputLayers)
            {
                foreach (HiddenOrOutputNeuron neuron in layer.Neurons)
                {
                    //For each layer and for each neuron, clear the cached activation value so it isn't used again
                    neuron.SetActivationToNull();
                }
            }
        }

        /// <summary>
        /// Calculates output of the neural network by calling <see cref="INeuron.CalculateActivation"/> on each output neuron.
        /// If <see cref="FeedForward"/> has not been called, this will recursively calculate the output by moving right to left in the network.
        /// </summary>
        private IEnumerable<double> GetOutput()
        {
            return OutputLayer.Neurons.Select(n => n.CalculateActivation());
        }

        /// <summary>
        /// Sets the output values of each neuron, moving from left to right.
        /// </summary>
        private void FeedForward()
        {
            foreach (HiddenOrOutputLayer layer in HiddenAndOutputLayers)
            {
                foreach (HiddenOrOutputNeuron neuron in layer.Neurons)
                {
                    neuron.CalculateActivation();
                }
            }
        }

        /// <summary>
        /// Sets the network input, feeds forward, and returns the output.
        /// </summary>
        public IEnumerable<double> Predict(IReadOnlyCollection<double> inputs)
        {
            SetInputLayerOutputsAndResetOtherOutputs(inputs);
            FeedForward();
            return GetOutput();
        }
    }
}
