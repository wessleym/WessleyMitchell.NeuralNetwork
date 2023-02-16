using System;
using System.Collections.Generic;
using System.Linq;
using WessleyMitchell.NeuralNetworking.Functions;

namespace WessleyMitchell.NeuralNetworking.Neurons
{
    public class HiddenOrOutputNeuron : INeuron
    {
        private readonly IActivationFunction activationFunction;
        private readonly IInputFunction inputFunction;

        /// <summary>
        /// Input <see cref="Synapse"/> connections
        /// </summary>
        private readonly List<Synapse> inputs = new List<Synapse>();

        private readonly List<Synapse> outputs = new List<Synapse>();
        /// <summary>
        /// Output <see cref="Synapse"/> connections
        /// </summary>
        public IEnumerable<Synapse> Outputs => outputs;

        /// <summary>
        /// Cached value of activation
        /// </summary>
        private Nullable<double> activationNullable;

        /// <summary>
        /// Neuron error.  Used in training.
        /// </summary>
        private double? error = null;
        internal double Error
        {
            get { if (error != null) { return error.Value; } throw new InvalidOperationException(nameof(Error) + " was null."); }
            set { error = value; }
        }

        /// <summary>
        /// Neuron bias.  Used in training.
        /// </summary>
        private double bias;

        /// <summary>
        /// Used to accumulate biases for future averaging
        /// </summary>
        private readonly List<double> biasDeltas;
        internal HiddenOrOutputNeuron(IActivationFunction activationFunction, IInputFunction inputFunction)
        {
            this.activationFunction = activationFunction;
            this.inputFunction = inputFunction;
            inputs = new List<Synapse>();
            outputs = new List<Synapse>();
            bias = 0;
            biasDeltas = new List<double>();
        }

        /// <summary>
        /// Add <see cref="Synapse"/> to <see cref="inputs"/>
        /// </summary>
        internal void AddInput(Synapse input)
        {
            inputs.Add(input);
        }

        /// <summary>
        /// Add <see cref="Synapse"/> to <see cref="outputs"/>
        /// </summary>
        public void AddOutput(Synapse output)
        {
            outputs.Add(output);
        }

        /// <summary>
        /// Calculates <see cref="inputFunction"/> and adds <see cref="bias"/>
        /// </summary>
        /// <returns></returns>
        private double GetInputPlusBias()
        {
            return inputFunction.CalculateInput(inputs) + bias;
        }

        /// <summary>
        /// Calculates the activation after summing inputs and applying the activation function (and possibly applying a bias)
        /// </summary>
        public double CalculateActivation()
        {
            if (activationNullable == null)
            {
                double input = GetInputPlusBias();
                activationNullable = activationFunction.CalculateOutput(input);
            }
            return activationNullable.Value;
        }

        /// <summary>
        /// Resets the activation in the event of new input.  This prevents a cached value from a previous input from affecting the next input.
        /// </summary>
        internal void SetActivationToNull()
        {
            activationNullable = null;
        }

        /// <summary>
        /// Calculates the derivative of the activation function.
        /// </summary>
        internal double CalculateDerivative()
        {
            return activationFunction.CalculateDerivative(GetInputPlusBias());
        }

        /// <summary>
        /// Clears <see cref="biasDeltas"/> and <see cref="Synapse.weightDeltas"/>
        /// </summary>
        internal void PrepareForTraining()
        {
            biasDeltas.Clear();
            foreach (Synapse synapse in inputs)
            {
                synapse.ClearWeightDeltas();
            }
        }

        /// <summary>
        /// Adds a value to <see cref="biasDeltas"/> for future averaging
        /// </summary>
        internal void AddToBiasDeltas(double biasDelta)
        {
            biasDeltas.Add(biasDelta);
        }

        /// <summary>
        /// Calculates new average <see cref="Synapse.Weight"/>.  Optionally modifies <see cref="bias"/>.
        /// </summary>
        /// <param name="trainBiases">Modify <see cref="bias"/></param>
        internal void CommitChangesAfterTraining(bool trainBiases)
        {
            if (trainBiases)
            {
                if (!biasDeltas.Any()) { throw new InvalidOperationException("No biases have been added."); }
                bias += biasDeltas.Sum() / biasDeltas.Count;
            }
            foreach (Synapse synapse in inputs)
            {
                synapse.CalculateNewWeight();
            }
        }
    }
}
