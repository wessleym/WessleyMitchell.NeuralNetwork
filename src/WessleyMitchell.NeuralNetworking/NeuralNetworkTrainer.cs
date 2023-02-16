using System;
using System.Collections.Generic;
using System.Linq;
using WessleyMitchell.NeuralNetworking.Collections;
using WessleyMitchell.NeuralNetworking.Functions;
using WessleyMitchell.NeuralNetworking.Layers;
using WessleyMitchell.NeuralNetworking.Neurons;

namespace WessleyMitchell.NeuralNetworking
{
    public class NeuralNetworkTrainer
    {
        private readonly FeedForwardNeuralNetwork neuralNetwork;
        private readonly double learningRate;
        private readonly IErrorFunction errorFunction;
        private readonly bool trainBiases;
        public NeuralNetworkTrainer(FeedForwardNeuralNetwork neuralNetwork, double learningRate, IErrorFunction errorFunction, bool trainBiases)
        {
            this.neuralNetwork = neuralNetwork;
            this.learningRate = learningRate;
            this.errorFunction = errorFunction;
            this.trainBiases = trainBiases;
        }

        /// <summary>
        /// Calculates squared error loss (SEL)
        /// </summary>
        private double CalculateCost(IEnumerable<double> outputs, IEnumerable<double> expectedOutputs)
        {
            return outputs
                .Zip(expectedOutputs, (actual, expected) => new { actual, expected })
                .Select(x => errorFunction.CalculateOutput(x.expected, x.actual))
                .Sum();
        }

        /// <summary>
        /// Computes error for the output layer.  Error is computed differently in the output layer than in hidden layers.
        /// </summary>
        private void ComputeErrorForOutputLayer(IEnumerable<double> expectedOutputs)
        {
            IEnumerable<HiddenOrOutputNeuron> lastLayerNeurons = neuralNetwork.OutputLayer.Neurons;
            foreach (var lastLayerNeuronAndExpectedOutput in lastLayerNeurons.Zip(expectedOutputs, (lastLayerNeuron, expectedOutput) => new { lastLayerNeuron, expectedOutput }))
            {
                //For each last layer neuron, set the neuron's error to the value of the derivative of the neuron's activation function, and the multiply it times (expected - computed).
                HiddenOrOutputNeuron lastLayerNeuron = lastLayerNeuronAndExpectedOutput.lastLayerNeuron;
                double expectedOutput = lastLayerNeuronAndExpectedOutput.expectedOutput;
                double computedOutput = lastLayerNeuron.CalculateActivation();
                double derivative = lastLayerNeuron.CalculateDerivative();
                lastLayerNeuron.Error = derivative * (expectedOutput - computedOutput);//Some say this second factor should be the derivative of the error function, which, for MSE, is the same thing as the difference of the two numbers.//errorFunction.CalculateDerivative(expectedOutput, computedOutput);
            }
        }

        /// <summary>
        /// Computes error for hidden layers.  Error is computed differently in hidden layers than in output layers.
        /// </summary>
        private void ComputeErrorForHiddenLayers()
        {
            IEnumerable<HiddenOrOutputLayer> hiddenLayersReversed = neuralNetwork.HiddenLayers.Reverse();
            foreach (HiddenOrOutputLayer hiddenLayer in hiddenLayersReversed)
            {
                foreach (HiddenOrOutputNeuron neuron in hiddenLayer.Neurons)
                {
                    //For each hidden layer and for each neuron,
                    //(1) calculate an error value by summing each output synapse's weight times the synapse's output neuron's error and then
                    //(2) multiply that error sum by the value of the derivative of the activation function.
                    double errorSum = neuron.Outputs.Sum(o => o.Weight * o.OutputNeuronError);
                    double derivative = neuron.CalculateDerivative();
                    neuron.Error = errorSum * derivative;
                }
            }
        }

        /// <summary>
        /// Adds a new value to <see cref="HiddenOrOutputNeuron.biasDeltas"/> at each <see cref="HiddenOrOutputNeuron"/> for future averaging
        /// </summary>
        private void CollectBiasesForHiddenLayersAndOutputLayer()
        {
            IEnumerable<HiddenOrOutputLayer> layers = neuralNetwork.HiddenAndOutputLayers;
            foreach (HiddenOrOutputLayer layer in layers)
            {
                foreach (HiddenOrOutputNeuron neuron in layer.Neurons)
                {
                    //For each hidden or output layer, compute a bias equal to the learning rate times the neuron's error.
                    neuron.AddToBiasDeltas(learningRate * neuron.Error);
                }
            }
        }

        /// <summary>
        /// Adds a new value to <see cref="Synapse.weightDeltas"/> at each <see cref="Synapse"/> for future averaging
        /// </summary>
        private void CollectWeightsForHiddenLayers()
        {
            foreach (ILayer layer in neuralNetwork)
            {
                foreach (INeuron neuron in layer.Neurons)
                {
                    double neuronOutput = neuron.CalculateActivation();
                    foreach (Synapse synapse in neuron.Outputs)
                    {
                        //For each layer, for each neuron, and for each synapse,
                        //compute a weight equal to the learning rate times the neuron's output times the synapse's output neuron's error.
                        synapse.AddToWeightDeltas(learningRate * neuronOutput * synapse.OutputNeuronError);
                    }
                }
            }
        }

        /// <summary>
        /// Prepare to store data for training (and clear old data)
        /// </summary>
        private void PrepareForTraining()
        {
            foreach (HiddenOrOutputNeuron neuron in neuralNetwork.HiddenAndOutputLayers.SelectMany(l => l.Neurons))
            {
                neuron.PrepareForTraining();
            }
        }

        /// <summary>
        /// Complete training process by committing changes (e.g., averaging weights)
        /// </summary>
        private void CommitChangesAfterTraining()
        {
            foreach (HiddenOrOutputNeuron neuron in neuralNetwork.HiddenAndOutputLayers.SelectMany(l => l.Neurons))
            {
                neuron.CommitChangesAfterTraining(trainBiases);
            }
        }

        /// <summary>
        /// Train neural network
        /// </summary>
        public void Train(IReadOnlyCollection<InputsAndOutputsItem> inputsAndOutputs, int epochs, Action<double>? costSumUpdate = null)
        {
            for (int i = 0; i < epochs; i++)
            {
                PrepareForTraining();
                double costSum = 0;
                foreach (InputsAndOutputsItem inputsAndOutputsItem in inputsAndOutputs)
                {
                    double[] outputs = neuralNetwork.Predict(inputsAndOutputsItem.Inputs).ToArray();
                    costSum += CalculateCost(outputs, inputsAndOutputsItem.Outputs);
                    ComputeErrorForOutputLayer(inputsAndOutputsItem.Outputs);
                    ComputeErrorForHiddenLayers();
                    CollectBiasesForHiddenLayersAndOutputLayer();
                    CollectWeightsForHiddenLayers();
                }
                if (costSumUpdate != null) { costSumUpdate(costSum); }
                CommitChangesAfterTraining();
            }
        }

        /// <summary>
        /// Train neural network and write cost sum to <see cref="Console"/>
        /// </summary>
        public void TrainAndWriteToConsole(IReadOnlyCollection<InputsAndOutputsItem> inputsAndOutputs, int epochs)
        {
            Console.Write("Cost Sum:  ");
            int cursorLeft = Console.CursorLeft;
            int previousNumberLength = 0;
            Train(inputsAndOutputs, epochs, costSum=>
            {
                //Move the cursor back to the left and write over previous result
                Console.CursorLeft = cursorLeft;
                string costSumString = costSum.ToString();
                Console.Write(costSumString.PadRight(previousNumberLength));
                previousNumberLength = costSumString.Length;
            });
            Console.WriteLine();
        }

        /// <summary>
        /// Test neural network against <paramref name="inputsAndOutputs"/> and get an average cost sum
        /// </summary>
        public double Test(IEnumerable<InputsAndOutputsItem> inputsAndOutputs)
        {
            int dataCount = 0;
            double costSum = inputsAndOutputs.Select(inputsAndOutputsItem =>
            {
                //For each input set and output set, calculate the output and compare it with the actual output to get a cost sum
                dataCount++;
                double[] outputs = neuralNetwork.Predict(inputsAndOutputsItem.Inputs).ToArray();
                return CalculateCost(outputs, inputsAndOutputsItem.Outputs);
            }).Sum();
            return costSum / dataCount;
        }
    }
}
