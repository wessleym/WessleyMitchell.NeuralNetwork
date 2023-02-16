using System.Collections.Generic;
using System.Linq;
using WessleyMitchell.NeuralNetworking.Neurons;

namespace WessleyMitchell.NeuralNetworking.Layers
{
    /// <summary>
    /// The base class of <see cref="InputLayer"/> and <see cref="HiddenOrOutputLayer"/>.
    /// It's only feature is a <see cref="Weights"/> property, which can be useful in debugging.
    /// </summary>
    public class LayerBase
    {
        private readonly IReadOnlyCollection<INeuron> neurons;
        internal LayerBase(IReadOnlyCollection<INeuron> neurons)
        {
            this.neurons = neurons;
        }

        /// <summary>
        /// Presents weights of <see cref="neurons"/> as a matrix.  Useful only for debugging.
        /// </summary>
        internal double[,] Weights
        {
            get
            {
                double[][] jaggedArray = neurons.Select(n => n.Outputs.Select(o => o.Weight).ToArray()).ToArray();
                double[,] multidimensionalArray = new double[jaggedArray.Length, jaggedArray[0].Length];
                for (int i = 0; i < jaggedArray.Length; i++)
                {
                    for (int j = 0; j < jaggedArray[i].Length; j++)
                    {
                        multidimensionalArray[i, j] = jaggedArray[i][j];
                    }
                }
                return multidimensionalArray;
            }
        }
    }
}
