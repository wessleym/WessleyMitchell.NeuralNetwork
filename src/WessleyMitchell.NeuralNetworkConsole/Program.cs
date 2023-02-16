using WessleyMitchell.NeuralNetworking;
using WessleyMitchell.NeuralNetworking.Collections;
using WessleyMitchell.NeuralNetworking.Functions;
using WessleyMitchell.NeuralNetworking.Layers;
using WessleyMitchell.NeuralNetworking.RandomNumberGenerators;

namespace WessleyMitchell.NeuralNetworkConsole
{
    internal class Program
    {
        static void Main()
        {
            XOR();
            Console.WriteLine();
            ReturnLastInput();
        }

        /// <summary>
        /// Trains a network to understand XOR
        /// </summary>
        private static void XOR()
        {
            Console.WriteLine(nameof(XOR));

            //Construct neural network
            FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(2, new RandomNumberGeneratorNegative1To1())
            {
                LayerFactory.ConstructHiddenOrOutputLayer(2, new SigmoidFunction(), new WeightedSumFunction()),
                LayerFactory.ConstructHiddenOrOutputLayer(1, new SigmoidFunction(), new WeightedSumFunction()),
            };

            //Construct trainer
            NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(network, 5, new MeanSquaredErrorFunction(), true);

            //Construct inputs and outputs
            InputsAndOutputsCollection inputsAndOutputs = new InputsAndOutputsCollection()
            {
                { new double[] { 0, 0 }, new double[] { 0 } },
                { new double[] { 0, 1 }, new double[] { 1 } },
                { new double[] { 1, 0 }, new double[] { 1 } },
                { new double[] { 1, 1 }, new double[] { 0 } }
            };

            //Data is not split into test and training sets because it is so limited with only four inputs and outputs.
            //Splitting it would mean the network would be trained on a subset of the possibilities.

            //Train based on all inputs and outputs
            trainer.TrainAndWriteToConsole(inputsAndOutputs, 10000);

            //Test based on all inputs and outputs and write error to console
            double error = trainer.Test(inputsAndOutputs);
            Console.WriteLine("Error After Training:  " + error);

            //Compute final answers and write them to console
            double result00 = network.Predict(new double[] { 0, 0 }).Single();
            double result01 = network.Predict(new double[] { 0, 1 }).Single();
            double result10 = network.Predict(new double[] { 1, 0 }).Single();
            double result11 = network.Predict(new double[] { 1, 1 }).Single();
            Console.WriteLine("Result (0, 0):  " + result00);
            Console.WriteLine("Result (0, 1):  " + result01);
            Console.WriteLine("Result (1, 0):  " + result10);
            Console.WriteLine("Result (1, 1):  " + result11);
        }

        /// <summary>
        /// Trains a network to ignore all but the last input
        /// </summary>
        private static void ReturnLastInput()
        {
            Console.WriteLine(nameof(ReturnLastInput));
            //Construct neural network
            FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(3, new RandomNumberGeneratorNegative_5To_5())
            {
                LayerFactory.ConstructHiddenOrOutputLayer(3, new RectifiedLinearUnitFunction(), new WeightedSumFunction()),
                LayerFactory.ConstructHiddenOrOutputLayer(1, new SigmoidFunction(), new WeightedSumFunction())
            };

            //Construct trainer
            NeuralNetworkTrainer trainer = new NeuralNetworkTrainer(network, .1, new MeanSquaredErrorFunction(), false);

            //Set up inputs and outputs
            InputsAndOutputsCollection inputsAndOutputs = new InputsAndOutputsCollection();
            Random random = new Random(0);//A seed of zero is unsed for a predictabily random set.
            for (int i = 0; i < 1000; i++)
            {
                int output = random.Next(0, 2);
                //Below, max values greater than ~25 result in the derivative of sigmoid being zero, which makes error zero, which makes training impossible.
                //This is because sigmoid's Math.Exp(-input) results in a very small number.
                const int min = 1, max = 25;
                inputsAndOutputs.Add(new double[] { random.Next(min, max), random.Next(min, max), output }, new double[] { output });
            }

            //Split inputs and outputs into training and test data
            IReadOnlyCollection<InputsAndOutputsItem> trainData, testData;
            inputsAndOutputs.Split(.3, out trainData, out testData);

            //Train based on training data
            trainer.TrainAndWriteToConsole(trainData, 3000);

            //Test based on test data and write error to console
            double error = trainer.Test(testData);
            Console.WriteLine("Error After Training:  " + error);

            //Compute final answers and write them to console
            var result0 = network.Predict(new double[] { 5, 10, 0 }).Single();
            var result1 = network.Predict(new double[] { 25, 2, 1 }).Single();
            Console.WriteLine("Result (5, 10, 0):  " + result0);
            Console.WriteLine("Result (25, 2, 1):  " + result1);
        }
    }
}