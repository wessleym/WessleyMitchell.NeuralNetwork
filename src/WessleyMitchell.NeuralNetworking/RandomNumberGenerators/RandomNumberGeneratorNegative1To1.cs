namespace WessleyMitchell.NeuralNetworking.RandomNumberGenerators
{
    /// <summary>
    /// Generates a random number from -1 to 1
    /// </summary>
    public class RandomNumberGeneratorNegative1To1 : RandomNumberGenerator
    {
        public RandomNumberGeneratorNegative1To1()
            : base(random => (random * 2) - 1)
        { }
    }
}
