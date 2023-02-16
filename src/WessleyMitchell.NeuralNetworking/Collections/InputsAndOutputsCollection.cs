using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace WessleyMitchell.NeuralNetworking.Collections
{
    public class InputsAndOutputsCollection : IReadOnlyCollection<InputsAndOutputsItem>
    {
        private readonly List<InputsAndOutputsItem> list;
        public InputsAndOutputsCollection()
        {
            list = new List<InputsAndOutputsItem>();
        }

        public IEnumerator<InputsAndOutputsItem> GetEnumerator()
        {
            return list.GetEnumerator();
        }
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public int Count => list.Count;

        public void Add(IReadOnlyCollection<double> inputs, IReadOnlyCollection<double> outputs)
        {
            list.Add(new InputsAndOutputsItem(inputs, outputs));
        }

        public void Split(double testRatio, out IReadOnlyCollection<InputsAndOutputsItem> train, out IReadOnlyCollection<InputsAndOutputsItem> test)
        {
            if (testRatio <= 0 || testRatio >= 1) { throw new ArgumentOutOfRangeException(nameof(testRatio)); }
            List<InputsAndOutputsItem> trainLocal = new List<InputsAndOutputsItem>(), testLocal = new List<InputsAndOutputsItem>();
            Random random = new Random();
            var randomizedOrder = this.OrderBy(_ => random.NextDouble()).ToArray();
            int testLowerBound = list.Count - (int)Math.Floor(list.Count * testRatio);
            for (int i = 0; i < list.Count; i++)
            {
                InputsAndOutputsItem inputsAndOutput = list[i];
                List<InputsAndOutputsItem> targetList = i < testLowerBound ? trainLocal : testLocal;
                targetList.Add(inputsAndOutput);
            }
            train = trainLocal;
            test = testLocal;
        }
    }
}
