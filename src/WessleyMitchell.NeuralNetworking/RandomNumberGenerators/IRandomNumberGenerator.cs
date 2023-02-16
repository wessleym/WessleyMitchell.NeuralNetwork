namespace WessleyMitchell.NeuralNetworking.RandomNumberGenerators
{
    /// <summary>
    /// Defines a simple interface for getting the next random <see cref="double"/>
    /// </summary>
    public interface IRandomNumberGenerator
    {
        /// <summary>
        /// Get next random <see cref="double"/>
        /// </summary>
        double Next();
    }
}
