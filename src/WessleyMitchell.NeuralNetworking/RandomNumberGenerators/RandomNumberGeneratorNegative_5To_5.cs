namespace WessleyMitchell.NeuralNetworking.RandomNumberGenerators
{
    /// <summary>
    /// Generates a random number from -0.5 to .5
    /// </summary>
    public class RandomNumberGeneratorNegative_5To_5 : RandomNumberGenerator
    {
        public RandomNumberGeneratorNegative_5To_5()
            : base(random => random - .5)
        { }
    }
}
