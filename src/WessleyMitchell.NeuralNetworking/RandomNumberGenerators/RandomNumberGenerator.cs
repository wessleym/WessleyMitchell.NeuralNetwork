using System;

namespace WessleyMitchell.NeuralNetworking.RandomNumberGenerators
{
    /// <summary>
    /// Allows for configurable random number generation
    /// </summary>
    public class RandomNumberGenerator : IRandomNumberGenerator
    {
        ///Use the same <see cref="Random"/> for each <see cref="RandomNumberGenerator"/>.  Also, use a seed of 0 when debugging for predictability.
        private static readonly Random random = new Random(
#if DEBUG
                0
#endif
                );
        private readonly Func<double, double> modifier;
        /// <summary>
        /// Constructs a <see cref="RandomNumberGenerator"/> with output modified by <paramref name="modifier"/>
        /// </summary>
        /// <param name="modifier">Specifies how each <see cref="double"/> will be modified before being returned when <see cref="Next"/> is called</param>
        public RandomNumberGenerator(Func<double, double> modifier)
        {
            this.modifier = modifier;
        }

        public double Next()
        {
            return modifier(random.NextDouble());
        }
    }
}
